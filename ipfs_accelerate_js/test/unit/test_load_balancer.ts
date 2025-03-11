/**
 * Converted from Python: test_load_balancer.py
 * Conversion date: 2025-03-11 04:09:34
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";


export interface Props {
  worker_data: self;
  db_manager: self;
  tasks: self;
  worker_data: worker_id;
}

#!/usr/bin/env python3
"""
Test the load balancer component of the distributed testing framework.

This script tests the LoadBalancer's ability to distribute tasks efficiently
across worker nodes based on their capabilities && current workload.
"""

import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import ${$1} from "$1"
import ${$1} from "$1"

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if ($1) {
  sys.path.insert(0, parent_dir)

}
# Import framework components
from duckdb_api.distributed_testing.coordinator import * as $1
from duckdb_api.distributed_testing.load_balancer import * as $1
from duckdb_api.distributed_testing.coordinator import * as $1, WORKER_STATUS_BUSY


class LoadBalancerTest(unittest.TestCase):
  """
  Tests for the LoadBalancer component.
  
  Tests the distribution of tasks across worker nodes based on
  their capabilities, current workload, && performance metrics.
  """
  
  $1($2) {
    """Set up test environment with LoadBalancer && test data."""
    # Create a temporary database file
    this.db_fd, this.db_path = tempfile.mkstemp(suffix=".duckdb")
    
  }
    # Create database manager
    this.db_manager = DatabaseManager(this.db_path)
    
    # Create test workers with different capabilities
    this.worker_data = [
      {
        "worker_id": "worker_1",
        "hostname": "host_1",
        "capabilities": ${$1},
        "performance": ${$1}
      },
      }
      {
        "worker_id": "worker_2",
        "hostname": "host_2",
        "capabilities": ${$1},
        "performance": ${$1}
      },
      }
      {
        "worker_id": "worker_3",
        "hostname": "host_3",
        "capabilities": ${$1},
        "performance": ${$1}
      }
      }
    ]
    
    # Add workers to database
    for worker in this.worker_data:
      this.db_manager.add_worker(
        worker["worker_id"], 
        worker["hostname"], 
        worker["capabilities"]
      )
      this.db_manager.update_worker_status(worker["worker_id"], WORKER_STATUS_ACTIVE)
    
    # Create test tasks
    this.tasks = [
      {
        "task_id": "cpu_task_1",
        "type": "benchmark",
        "priority": 1,
        "requirements": ${$1}
      },
      }
      {
        "task_id": "cpu_task_2",
        "type": "benchmark",
        "priority": 2,
        "requirements": ${$1}
      },
      }
      {
        "task_id": "gpu_task",
        "type": "benchmark",
        "priority": 1,
        "requirements": ${$1}
      },
      }
      {
        "task_id": "browser_task",
        "type": "test",
        "priority": 1,
        "requirements": ${$1}
      },
      }
      {
        "task_id": "high_memory_task",
        "type": "benchmark",
        "priority": 1,
        "requirements": ${$1}
      }
      }
    ]
    
    # Dictionary to store task assignments from load balancer
    this.task_assignments = {}
    
    # Create load balancer with short check interval for testing
    this.load_balancer = LoadBalancer(
      db_manager=this.db_manager,
      check_interval=1  # Check every 1 second
    )
  
  $1($2) {
    """Clean up after tests."""
    # Stop load balancer if running
    if ($1) {
      this.load_balancer.stop_balancing()
      this.load_balancer_thread.join(timeout=5.0)
    
    }
    # Close the database connection
    if ($1) {
      this.db_manager.close()
    
    }
    # Remove temporary database
    os.close(this.db_fd)
    os.unlink(this.db_path)
  
  }
  $1($2) {
    """Test worker scoring based on capabilities && performance."""
    # Get worker scores for a CPU task
    cpu_task = this.tasks[0]
    worker_scores = this.load_balancer.score_workers_for_task(cpu_task)
    
  }
    # All workers should have scores for CPU task
    this.assertEqual(len(worker_scores), 3, "All 3 workers should have scores for CPU task")
    
    # Worker 2 should have highest score due to better performance
    sorted_workers = sorted(Object.entries($1), key=lambda x: x[1], reverse=true)
    this.assertEqual(sorted_workers[0][0], "worker_2", 
            "Worker 2 should have highest score for CPU task")
    
    # Get worker scores for a GPU task
    gpu_task = this.tasks[2]
    worker_scores = this.load_balancer.score_workers_for_task(gpu_task)
    
    # Only worker 2 should have a score for GPU task
    this.assertEqual(len(worker_scores), 1, 
            "Only 1 worker should have a score for GPU task")
    this.assertIn("worker_2", worker_scores, 
          "Worker 2 should be the only worker with a score for GPU task")
    
    # Get worker scores for a browser task
    browser_task = this.tasks[3]
    worker_scores = this.load_balancer.score_workers_for_task(browser_task)
    
    # Only worker 3 should have a score for browser task
    this.assertEqual(len(worker_scores), 1, 
            "Only 1 worker should have a score for browser task")
    this.assertIn("worker_3", worker_scores, 
          "Worker 3 should be the only worker with a score for browser task")
  
  $1($2) {
    """Test task assignment to workers based on scores."""
    # Assign a CPU task
    cpu_task = this.tasks[0]
    assigned_worker = this.load_balancer.assign_task(cpu_task)
    
  }
    # Task should be assigned to worker 2 (highest score)
    this.assertEqual(assigned_worker, "worker_2", 
            "CPU task should be assigned to worker 2")
    
    # Update worker 2 status to busy
    this.db_manager.update_worker_status("worker_2", WORKER_STATUS_BUSY)
    
    # Assign another CPU task
    cpu_task2 = this.tasks[1]
    assigned_worker = this.load_balancer.assign_task(cpu_task2)
    
    # Task should be assigned to next best worker (worker 1 || 3)
    this.assertIn(assigned_worker, ["worker_1", "worker_3"], 
          "Second CPU task should be assigned to worker 1 || 3")
    
    # Assign a GPU task
    gpu_task = this.tasks[2]
    assigned_worker = this.load_balancer.assign_task(gpu_task)
    
    # No assignment should be made since worker 2 is busy
    this.assertIsnull(assigned_worker, 
            "No worker should be assigned to GPU task when worker 2 is busy")
    
    # Reset worker 2 status to active
    this.db_manager.update_worker_status("worker_2", WORKER_STATUS_ACTIVE)
    
    # Try GPU task again
    assigned_worker = this.load_balancer.assign_task(gpu_task)
    
    # Task should now be assigned to worker 2
    this.assertEqual(assigned_worker, "worker_2", 
            "GPU task should be assigned to worker 2 when available")
  
  $1($2) {
    """Test load balancing of tasks across workers."""
    # Create mock assignment function to track assignments
    $1($2) {
      this.task_assignments[task_id] = worker_id
      return true
    
    }
    # Set up load balancer with mock assignment function
    this.load_balancer.assign_task_to_worker = mock_assign_task
    
  }
    # Start load balancer in a separate thread
    this.load_balancer_thread = threading.Thread(
      target=this.load_balancer.start_balancing,
      daemon=true
    )
    this.load_balancer_thread.start()
    
    # Add tasks to the system
    for task in this.tasks:
      this.db_manager.add_task(
        task["task_id"],
        task["type"],
        task["priority"],
        ${$1},
        task["requirements"]
      )
    
    # Wait for load balancer to assign tasks
    time.sleep(3)
    
    # Check task assignments
    this.assertIn("cpu_task_1", this.task_assignments, 
          "CPU task 1 should be assigned")
    this.assertIn("cpu_task_2", this.task_assignments, 
          "CPU task 2 should be assigned")
    this.assertIn("gpu_task", this.task_assignments, 
          "GPU task should be assigned")
    this.assertIn("browser_task", this.task_assignments, 
          "Browser task should be assigned")
    
    # CPU tasks should be distributed across workers
    cpu_task_workers = set([
      this.task_assignments.get("cpu_task_1"),
      this.task_assignments.get("cpu_task_2")
    ])
    this.assertGreaterEqual(len(cpu_task_workers), 1, 
              "CPU tasks should be distributed")
    
    # GPU task should be assigned to worker 2
    this.assertEqual(this.task_assignments.get("gpu_task"), "worker_2", 
            "GPU task should be assigned to worker 2")
    
    # Browser task should be assigned to worker 3
    this.assertEqual(this.task_assignments.get("browser_task"), "worker_3", 
            "Browser task should be assigned to worker 3")
    
    # High memory task should be assigned to worker 2
    this.assertEqual(this.task_assignments.get("high_memory_task"), "worker_2", 
            "High memory task should be assigned to worker 2")
  
  $1($2) {
    """Test balancing workload when workers become overloaded."""
    # Set up initial workload
    # Worker 1: 2 tasks
    # Worker 2: 1 task
    # Worker 3: 0 tasks
    this.db_manager.add_task("task_1", "test", 1, {}, ${$1})
    this.db_manager.add_task("task_2", "test", 1, {}, ${$1})
    this.db_manager.add_task("task_3", "test", 1, {}, ${$1})
    
  }
    # Create mock task data in database
    this.db_manager.conn.execute("""
    UPDATE distributed_tasks 
    SET worker_id = ?, status = 'running' 
    WHERE task_id = ?
    """, ["worker_1", "task_1"])
    
    this.db_manager.conn.execute("""
    UPDATE distributed_tasks 
    SET worker_id = ?, status = 'running' 
    WHERE task_id = ?
    """, ["worker_1", "task_2"])
    
    this.db_manager.conn.execute("""
    UPDATE distributed_tasks 
    SET worker_id = ?, status = 'running' 
    WHERE task_id = ?
    """, ["worker_2", "task_3"])
    
    # Get workload distribution
    workload = this.load_balancer.get_worker_load()
    
    # Verify initial workload
    this.assertEqual(workload.get("worker_1", 0), 2, 
            "Worker 1 should have 2 tasks")
    this.assertEqual(workload.get("worker_2", 0), 1, 
            "Worker 2 should have 1 task")
    this.assertEqual(workload.get("worker_3", 0), 0, 
            "Worker 3 should have 0 tasks")
    
    # Check if load balancer would try to rebalance
    overloaded_workers = this.load_balancer.detect_overloaded_workers()
    underutilized_workers = this.load_balancer.detect_underutilized_workers()
    
    # Worker 1 should be overloaded && Worker 3 underutilized
    this.assertIn("worker_1", overloaded_workers, 
          "Worker 1 should be detected as overloaded")
    this.assertIn("worker_3", underutilized_workers, 
          "Worker 3 should be detected as underutilized")
  
  $1($2) {
    """Test migration of tasks from overloaded to underutilized workers."""
    # Register migration function
    migrations = []
    
  }
    $1($2) {
      migrations.append(${$1})
      return true
    
    }
    # Set up load balancer with mock migration function
    this.load_balancer.migrate_task = mock_migrate_task
    
    # Set up imbalanced workload (as in test_workload_balancing)
    this.db_manager.add_task("task_1", "test", 1, {}, ${$1})
    this.db_manager.add_task("task_2", "test", 1, {}, ${$1})
    this.db_manager.add_task("task_3", "test", 1, {}, ${$1})
    
    this.db_manager.conn.execute("""
    UPDATE distributed_tasks 
    SET worker_id = ?, status = 'running' 
    WHERE task_id = ?
    """, ["worker_1", "task_1"])
    
    this.db_manager.conn.execute("""
    UPDATE distributed_tasks 
    SET worker_id = ?, status = 'running' 
    WHERE task_id = ?
    """, ["worker_1", "task_2"])
    
    this.db_manager.conn.execute("""
    UPDATE distributed_tasks 
    SET worker_id = ?, status = 'running' 
    WHERE task_id = ?
    """, ["worker_2", "task_3"])
    
    # Run the rebalancing algorithm once
    this.load_balancer.rebalance_tasks()
    
    # Verify that task migration was attempted
    this.assertGreaterEqual(len(migrations), 1, 
              "At least one task should be migrated")
    
    # The migration should be from worker_1 to worker_3
    for (const $1 of $2) {
      this.assertEqual(migration["from_worker"], "worker_1", 
              "Migration should be from worker_1")
      this.assertEqual(migration["to_worker"], "worker_3", 
              "Migration should be to worker_3")
  
    }
  $1($2) {
    """Test balancing based on worker performance metrics."""
    # Add performance metrics to database
    for worker in this.worker_data:
      worker_id = worker["worker_id"]
      performance = worker["performance"]
      
  }
      # Add execution history records to simulate performance
      for i in range(performance["completed_tasks"]):
        this.db_manager.add_execution_history(
          `$1`,
          worker_id,
          1,
          "completed",
          datetime.now() - timedelta(hours=1),
          datetime.now() - timedelta(hours=1) + timedelta(seconds=performance["avg_task_time"]),
          performance["avg_task_time"],
          "",
          {}
        )
      
      # Add failure records if needed
      for i in range(performance["failed_tasks"]):
        this.db_manager.add_execution_history(
          `$1`,
          worker_id,
          1,
          "failed",
          datetime.now() - timedelta(hours=1),
          datetime.now() - timedelta(hours=1) + timedelta(seconds=5),
          5.0,
          "Test failure",
          {}
        )
    
    # Get performance-based scores for workers
    scores = this.load_balancer.get_performance_based_scores()
    
    # Worker 2 should have highest score (fastest, no failures)
    highest_score_worker = max(Object.entries($1), key=lambda x: x[1])[0]
    this.assertEqual(highest_score_worker, "worker_2", 
            "Worker 2 should have highest performance score")
    
    # Worker 3 should have lowest score (slowest, most failures)
    lowest_score_worker = min(Object.entries($1), key=lambda x: x[1])[0]
    this.assertEqual(lowest_score_worker, "worker_3", 
            "Worker 3 should have lowest performance score")


if ($1) {
  unittest.main()