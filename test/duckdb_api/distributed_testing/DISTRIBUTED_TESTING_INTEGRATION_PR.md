# Distributed Testing Framework Integration Plan

## Overview

This document outlines the comprehensive plan for advancing the Distributed Testing Framework to fully integrate it with the existing test infrastructure and enhance its capabilities. The framework currently has several key components implemented but requires additional integration work to function as a complete end-to-end system.

## Current Status

- Core coordinator and worker infrastructure is in place
- Error visualization and handling systems are complete
- Load balancer and task scheduler foundations are established
- Monitoring dashboard has been developed
- Hardware taxonomy and capability detection systems are implemented

## Key Integration Goals

1. **Worker-DuckDB Integration**: Connect worker nodes directly to the DuckDB database system for efficient test result storage
2. **Test Generator Integration**: Enable the framework to work with template-based test generators
3. **Enhanced JWT Authentication**: Implement comprehensive security with properly scoped JWT tokens
4. **Cross-Platform Worker Support**: Ensure workers can run on diverse operating systems and container environments
5. **CI/CD Pipeline Integration**: Complete integration with GitHub Actions for automated distributed testing
6. **Dynamic Resource Management**: Improve resource allocation and deallocation with adaptive scaling

## Implementation Plan

### Phase 1: DuckDB Integration Enhancement

#### Tasks:
- [x] Create DuckDBResultProcessor class to handle direct worker-to-database result storage
- [x] Implement batch result insertion for efficiency
- [x] Add database connection pooling for high-concurrency environments
- [x] Create result validation schema to ensure data integrity
- [x] Develop transactional result storage with rollback capability

#### Implementation Details:
```python
class DuckDBResultProcessor:
    """Processes test results and stores them directly in DuckDB."""
    
    def __init__(self, db_path, pool_size=5):
        """Initialize with database path and connection pool size."""
        self.db_path = db_path
        self.pool_size = pool_size
        self.connection_pool = []
        self.pool_lock = threading.Lock()
        self.initialize_pool()
    
    def initialize_pool(self):
        """Create connection pool for database access."""
        with self.pool_lock:
            for _ in range(self.pool_size):
                conn = duckdb.connect(self.db_path)
                self.connection_pool.append(conn)
    
    def get_connection(self):
        """Get a connection from the pool."""
        with self.pool_lock:
            if not self.connection_pool:
                return duckdb.connect(self.db_path)
            return self.connection_pool.pop()
    
    def release_connection(self, conn):
        """Return a connection to the pool."""
        with self.pool_lock:
            self.connection_pool.append(conn)
    
    def store_result(self, result_data):
        """Store a single test result in the database."""
        conn = self.get_connection()
        try:
            # Validate result data
            self._validate_result(result_data)
            
            # Convert to appropriate format
            db_record = self._convert_to_db_format(result_data)
            
            # Store in database with transaction support
            conn.execute("BEGIN TRANSACTION")
            conn.execute("""
                INSERT INTO test_results (
                    test_id, worker_id, model_name, hardware_type, 
                    execution_time, success, error_message, timestamp,
                    memory_usage, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                db_record['test_id'], db_record['worker_id'],
                db_record['model_name'], db_record['hardware_type'],
                db_record['execution_time'], db_record['success'],
                db_record['error_message'], db_record['timestamp'],
                db_record['memory_usage'], json.dumps(db_record['details'])
            ])
            conn.execute("COMMIT")
            return True
        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error(f"Error storing result: {e}")
            return False
        finally:
            self.release_connection(conn)
    
    def store_batch_results(self, results):
        """Store multiple test results efficiently."""
        conn = self.get_connection()
        try:
            conn.execute("BEGIN TRANSACTION")
            for result in results:
                self._validate_result(result)
                db_record = self._convert_to_db_format(result)
                conn.execute("""
                    INSERT INTO test_results (
                        test_id, worker_id, model_name, hardware_type, 
                        execution_time, success, error_message, timestamp,
                        memory_usage, details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    db_record['test_id'], db_record['worker_id'],
                    db_record['model_name'], db_record['hardware_type'],
                    db_record['execution_time'], db_record['success'],
                    db_record['error_message'], db_record['timestamp'],
                    db_record['memory_usage'], json.dumps(db_record['details'])
                ])
            conn.execute("COMMIT")
            return True
        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error(f"Error storing batch results: {e}")
            return False
        finally:
            self.release_connection(conn)
    
    def _validate_result(self, result):
        """Validate result data structure and types."""
        required_fields = ['test_id', 'worker_id', 'model_name', 'hardware_type', 'success']
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
    
    def _convert_to_db_format(self, result):
        """Convert result to database-compatible format."""
        db_record = {
            'test_id': result.get('test_id'),
            'worker_id': result.get('worker_id'),
            'model_name': result.get('model_name'),
            'hardware_type': result.get('hardware_type'),
            'execution_time': result.get('execution_time', 0.0),
            'success': result.get('success', False),
            'error_message': result.get('error_message', ''),
            'timestamp': result.get('timestamp', datetime.now().isoformat()),
            'memory_usage': result.get('memory_usage', 0.0),
            'details': result.get('details', {})
        }
        return db_record
```

### Phase 2: Template-Based Test Generator Integration

#### Tasks:
- [x] Create TestGeneratorIntegration class to connect with template generators
- [x] Implement model-to-test conversion for dynamic test generation
- [x] Add support for hardware-specific test template selection
- [x] Develop task dependency tracking for generated test suites
- [x] Create template database system for storing and retrieving templates

#### Implementation Details:
```python
class TestGeneratorIntegration:
    """Integrates with template-based test generators for dynamic test creation."""
    
    def __init__(self, template_db_path, coordinator_url):
        """Initialize with template database path and coordinator URL."""
        self.template_db_path = template_db_path
        self.coordinator_url = coordinator_url
        self.template_db = duckdb.connect(template_db_path)
        self.coordinator_client = CoordinatorClient(coordinator_url)
    
    def generate_and_submit_tests(self, model_name, hardware_types=None, batch_sizes=None):
        """Generate tests from templates and submit to the coordinator."""
        # Determine model family
        model_family = self._get_model_family(model_name)
        
        # Find appropriate templates
        templates = self._fetch_templates(model_family, hardware_types)
        
        # Generate tests
        generated_tests = []
        for template in templates:
            for hardware in hardware_types or ["cpu"]:
                for batch_size in batch_sizes or [1, 4]:
                    test_config = {
                        "model_name": model_name,
                        "model_family": model_family,
                        "hardware_type": hardware,
                        "batch_size": batch_size,
                        "template_id": template["template_id"]
                    }
                    test = self._generate_test(template, test_config)
                    generated_tests.append(test)
        
        # Set up dependencies between tests
        tests_with_dependencies = self._setup_dependencies(generated_tests)
        
        # Submit to coordinator
        submission_results = []
        for test in tests_with_dependencies:
            result = self.coordinator_client.submit_task(test)
            submission_results.append(result)
        
        return submission_results
    
    def _get_model_family(self, model_name):
        """Determine the model family from the model name."""
        # Query the template database to find model family
        result = self.template_db.execute(f"""
            SELECT model_family FROM model_mapping 
            WHERE model_name = '{model_name}'
        """).fetchone()
        
        if result:
            return result[0]
            
        # Try to infer from name if not found
        if "bert" in model_name.lower():
            return "text_embedding"
        elif "t5" in model_name.lower():
            return "text_generation"
        elif "vit" in model_name.lower():
            return "vision"
        elif "whisper" in model_name.lower():
            return "audio"
        else:
            return "unknown"
    
    def _fetch_templates(self, model_family, hardware_types=None):
        """Fetch appropriate templates for the model family and hardware types."""
        query = f"""
            SELECT * FROM templates 
            WHERE model_family = '{model_family}'
        """
        
        if hardware_types:
            hardware_list = ', '.join([f"'{h}'" for h in hardware_types])
            query += f" AND hardware_type IN ({hardware_list})"
            
        return self.template_db.execute(query).fetchall()
    
    def _generate_test(self, template, config):
        """Generate a test from a template and configuration."""
        # Extract template content
        template_content = template["content"]
        
        # Perform variable substitution
        for key, value in config.items():
            template_content = template_content.replace(f"${{{key}}}", str(value))
        
        # Create task definition
        task = {
            "test_id": str(uuid.uuid4()),
            "model_name": config["model_name"],
            "model_family": config["model_family"],
            "hardware_type": config["hardware_type"],
            "batch_size": config["batch_size"],
            "test_content": template_content,
            "test_type": "generated",
            "priority": self._calculate_priority(config),
            "requirements": {
                "hardware_type": config["hardware_type"],
                "min_memory_gb": self._estimate_memory_requirement(config),
                "test_timeout_seconds": 600  # 10 minutes default timeout
            }
        }
        
        return task
    
    def _setup_dependencies(self, tests):
        """Set up dependencies between tests for proper execution order."""
        # Group by model and hardware
        tests_by_group = {}
        for test in tests:
            key = (test["model_name"], test["hardware_type"])
            if key not in tests_by_group:
                tests_by_group[key] = []
            tests_by_group[key].append(test)
        
        # Set up dependencies within each group
        result = []
        for group_tests in tests_by_group.values():
            # Sort by batch size (ascending)
            sorted_tests = sorted(group_tests, key=lambda t: t["batch_size"])
            
            # Set up dependencies (each test depends on the previous one)
            for i in range(1, len(sorted_tests)):
                if "dependencies" not in sorted_tests[i]:
                    sorted_tests[i]["dependencies"] = []
                sorted_tests[i]["dependencies"].append(sorted_tests[i-1]["test_id"])
            
            result.extend(sorted_tests)
        
        return result
    
    def _calculate_priority(self, config):
        """Calculate task priority based on configuration."""
        base_priority = 5  # Default priority
        
        # Adjust based on hardware type
        if config["hardware_type"] == "cpu":
            base_priority += 1  # Slightly higher priority for CPU tests
        elif config["hardware_type"] in ["cuda", "rocm"]:
            base_priority -= 1  # Slightly lower for GPU tests (more resource intensive)
        
        # Adjust based on batch size
        if config["batch_size"] <= 1:
            base_priority += 1  # Higher priority for small batch sizes
        elif config["batch_size"] >= 16:
            base_priority -= 1  # Lower priority for large batch sizes
            
        return max(1, min(10, base_priority))  # Ensure between 1-10
    
    def _estimate_memory_requirement(self, config):
        """Estimate memory requirement based on configuration."""
        base_memory = 1.0  # Default 1GB
        
        # Adjust based on model family
        if config["model_family"] == "text_generation":
            base_memory = 4.0
        elif config["model_family"] == "vision":
            base_memory = 2.0
            
        # Adjust based on batch size
        batch_factor = config["batch_size"] / 4.0  # Normalized to batch size 4
        memory_estimate = base_memory * batch_factor
        
        return max(0.5, memory_estimate)  # Minimum 0.5GB
```

### Phase 3: Enhanced JWT Authentication

#### Tasks:
- [ ] Implement proper JWT token generation with role-based access control
- [ ] Add token refresh mechanisms for long-running workers
- [ ] Create token revocation for security incidents
- [ ] Add fine-grained permission model for worker operations
- [ ] Implement secure token storage on worker nodes

#### Implementation Details:
```python
class EnhancedJWTAuthHandler:
    """Handles JWT authentication with enhanced security features."""
    
    def __init__(self, secret_key, token_expiry_minutes=60):
        """Initialize with secret key and token expiry time."""
        self.secret_key = secret_key
        self.token_expiry_minutes = token_expiry_minutes
        self.revoked_tokens = set()
        self.refresh_tokens = {}  # worker_id -> refresh_token
        
        # Define permission levels
        self.permission_levels = {
            "admin": ["read", "write", "manage_workers", "manage_tasks", "system_config"],
            "manager": ["read", "write", "manage_workers", "manage_tasks"],
            "worker": ["read", "write", "report_status"],
            "reader": ["read"]
        }
    
    def generate_token(self, worker_id, role="worker", additional_claims=None):
        """Generate a JWT token with claims based on worker role."""
        now = datetime.utcnow()
        expiry = now + timedelta(minutes=self.token_expiry_minutes)
        
        # Get permissions for the role
        permissions = self.permission_levels.get(role, ["read"])
        
        # Build claims
        claims = {
            "sub": worker_id,
            "iat": now,
            "exp": expiry,
            "role": role,
            "permissions": permissions
        }
        
        # Add any additional claims
        if additional_claims:
            claims.update(additional_claims)
            
        # Generate token
        token = jwt.encode(claims, self.secret_key, algorithm="HS256")
        
        # Generate refresh token for workers
        if role == "worker":
            refresh_token = self._generate_refresh_token(worker_id)
            self.refresh_tokens[worker_id] = refresh_token
            return token, refresh_token
            
        return token, None
    
    def _generate_refresh_token(self, worker_id):
        """Generate a refresh token for a worker."""
        now = datetime.utcnow()
        expiry = now + timedelta(days=7)  # Refresh tokens valid for 7 days
        
        claims = {
            "sub": worker_id,
            "iat": now,
            "exp": expiry,
            "type": "refresh"
        }
        
        return jwt.encode(claims, self.secret_key, algorithm="HS256")
    
    def verify_token(self, token):
        """Verify a JWT token and return the claims if valid."""
        if token in self.revoked_tokens:
            raise ValueError("Token has been revoked")
            
        try:
            claims = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return claims
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
    
    def refresh_access_token(self, refresh_token, worker_id):
        """Generate a new access token using a refresh token."""
        try:
            # Verify the refresh token
            claims = jwt.decode(refresh_token, self.secret_key, algorithms=["HS256"])
            
            # Check if it's a refresh token
            if claims.get("type") != "refresh":
                raise ValueError("Not a refresh token")
                
            # Check if it matches the worker
            if claims.get("sub") != worker_id:
                raise ValueError("Worker ID mismatch")
                
            # Check if it's the current refresh token
            if self.refresh_tokens.get(worker_id) != refresh_token:
                raise ValueError("Refresh token has been superseded")
                
            # Generate new access token
            role = "worker"  # Default role for refreshed tokens
            new_token, _ = self.generate_token(worker_id, role)
            
            return new_token
            
        except jwt.ExpiredSignatureError:
            raise ValueError("Refresh token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid refresh token")
    
    def revoke_token(self, token):
        """Revoke a token so it can no longer be used."""
        self.revoked_tokens.add(token)
        
        # Try to extract worker_id from token
        try:
            claims = jwt.decode(token, self.secret_key, algorithms=["HS256"], options={"verify_signature": True})
            worker_id = claims.get("sub")
            
            # If it's a worker, also invalidate their refresh token
            if worker_id and claims.get("role") == "worker":
                if worker_id in self.refresh_tokens:
                    self.refresh_tokens.pop(worker_id)
        except:
            # If we can't decode, just revoke the specific token
            pass
            
        return True
    
    def revoke_worker_tokens(self, worker_id):
        """Revoke all tokens for a specific worker."""
        # Remove refresh token
        if worker_id in self.refresh_tokens:
            self.refresh_tokens.pop(worker_id)
            
        # Note: Can't directly revoke access tokens since we don't store them by worker_id
        # The worker will need to re-authenticate
        
        return True
    
    def has_permission(self, token, required_permission):
        """Check if a token has the required permission."""
        try:
            claims = self.verify_token(token)
            permissions = claims.get("permissions", [])
            return required_permission in permissions
        except:
            return False
```

### Phase 4: Cross-Platform Worker Support

#### Tasks:
- [x] Enhance worker detection to support multiple operating systems
- [x] Create container-compatible worker deployment scripts
- [x] Implement environment-aware resource management
- [x] Add platform-specific hardware detection modules
- [x] Create unified worker interface for different platforms

#### Implementation Details:
```python
class CrossPlatformWorkerSupport:
    """Provides cross-platform support for worker deployment and management."""
    
    def __init__(self):
        """Initialize the cross-platform worker support."""
        self.platform_handlers = {
            "linux": LinuxPlatformHandler(),
            "windows": WindowsPlatformHandler(),
            "darwin": MacOSPlatformHandler(),
            "container": ContainerPlatformHandler()
        }
        
        self.current_platform = self._detect_platform()
        self.handler = self.platform_handlers.get(self.current_platform)
        
    def _detect_platform(self):
        """Detect the current platform."""
        if os.environ.get("CONTAINER_ENV"):
            return "container"
            
        return platform.system().lower()
    
    def get_worker_command(self, config):
        """Get the platform-specific command to start a worker."""
        if not self.handler:
            raise ValueError(f"Unsupported platform: {self.current_platform}")
            
        return self.handler.get_worker_command(config)
    
    def create_deployment_script(self, config, output_path):
        """Create a platform-specific deployment script."""
        if not self.handler:
            raise ValueError(f"Unsupported platform: {self.current_platform}")
            
        return self.handler.create_deployment_script(config, output_path)
    
    def install_dependencies(self, dependencies=None):
        """Install platform-specific dependencies."""
        if not self.handler:
            raise ValueError(f"Unsupported platform: {self.current_platform}")
            
        return self.handler.install_dependencies(dependencies)
    
    def detect_hardware(self):
        """Detect hardware capabilities in a platform-specific way."""
        if not self.handler:
            raise ValueError(f"Unsupported platform: {self.current_platform}")
            
        return self.handler.detect_hardware()
    
    def get_startup_script(self, coordinator_url, api_key, worker_id=None):
        """Generate a platform-specific worker startup script."""
        if not self.handler:
            raise ValueError(f"Unsupported platform: {self.current_platform}")
            
        return self.handler.get_startup_script(coordinator_url, api_key, worker_id)


class PlatformHandler:
    """Base class for platform-specific handlers."""
    
    def get_worker_command(self, config):
        """Get the command to start a worker."""
        raise NotImplementedError()
    
    def create_deployment_script(self, config, output_path):
        """Create a deployment script."""
        raise NotImplementedError()
    
    def install_dependencies(self, dependencies=None):
        """Install dependencies."""
        raise NotImplementedError()
    
    def detect_hardware(self):
        """Detect hardware capabilities."""
        raise NotImplementedError()
    
    def get_startup_script(self, coordinator_url, api_key, worker_id=None):
        """Generate a worker startup script."""
        raise NotImplementedError()


class LinuxPlatformHandler(PlatformHandler):
    """Linux-specific platform handler."""
    
    def get_worker_command(self, config):
        """Get the command to start a worker on Linux."""
        cmd = [
            "python3", 
            "run_worker_client.py",
            "--coordinator", config.get("coordinator_url", "http://localhost:8080"),
            "--api-key", config.get("api_key", "default_key")
        ]
        
        if "worker_id" in config:
            cmd.extend(["--worker-id", config["worker_id"]])
            
        if config.get("log_to_file"):
            cmd.extend(["--log-file", f"worker_{config.get('worker_id', 'unknown')}.log"])
            
        return " ".join(cmd)
    
    def create_deployment_script(self, config, output_path):
        """Create a Linux deployment script."""
        script_content = """#!/bin/bash
# Worker deployment script for Linux

# Configuration
COORDINATOR_URL="{coordinator_url}"
API_KEY="{api_key}"
WORKER_ID="{worker_id}"
LOG_FILE="worker_$WORKER_ID.log"

# Ensure dependencies are installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Attempting to install..."
    apt-get update && apt-get install -y python3 python3-pip || \
    yum install -y python3 python3-pip || \
    dnf install -y python3 python3-pip
fi

# Install Python dependencies
pip3 install -r requirements.txt

# Start the worker
python3 run_worker_client.py \\
    --coordinator "$COORDINATOR_URL" \\
    --api-key "$API_KEY" \\
    --worker-id "$WORKER_ID" \\
    --log-file "$LOG_FILE"
""".format(
            coordinator_url=config.get("coordinator_url", "http://localhost:8080"),
            api_key=config.get("api_key", "default_key"),
            worker_id=config.get("worker_id", "worker_" + str(uuid.uuid4())[:8])
        )
        
        with open(output_path, "w") as f:
            f.write(script_content)
            
        os.chmod(output_path, 0o755)  # Make executable
        return output_path
    
    def install_dependencies(self, dependencies=None):
        """Install dependencies on Linux."""
        if dependencies is None:
            dependencies = ["websockets", "psutil", "pyjwt"]
            
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + dependencies, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def detect_hardware(self):
        """Detect hardware capabilities on Linux."""
        hardware_info = {
            "platform": "linux",
            "cpu": self._detect_linux_cpu(),
            "memory": self._detect_linux_memory(),
            "gpu": self._detect_linux_gpu()
        }
        return hardware_info
    
    def _detect_linux_cpu(self):
        """Detect CPU information on Linux."""
        cpu_info = {
            "cores": os.cpu_count(),
            "model": "Unknown"
        }
        
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_info["model"] = line.split(":", 1)[1].strip()
                        break
        except:
            pass
            
        return cpu_info
    
    def _detect_linux_memory(self):
        """Detect memory information on Linux."""
        memory_info = {
            "total_gb": 0
        }
        
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        mem_kb = int(line.split()[1])
                        memory_info["total_gb"] = round(mem_kb / (1024 * 1024), 2)
                        break
        except:
            pass
            
        return memory_info
    
    def _detect_linux_gpu(self):
        """Detect GPU information on Linux."""
        gpu_info = {
            "count": 0,
            "devices": []
        }
        
        # Try nvidia-smi for NVIDIA GPUs
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                for i, line in enumerate(lines):
                    if not line.strip():
                        continue
                    parts = line.split(",")
                    if len(parts) >= 2:
                        gpu_info["devices"].append({
                            "id": i,
                            "name": parts[0].strip(),
                            "memory": parts[1].strip(),
                            "type": "cuda"
                        })
                gpu_info["count"] = len(gpu_info["devices"])
        except:
            pass
            
        # Try rocm-smi for AMD GPUs
        if gpu_info["count"] == 0:
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showproductname"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    gpu_names = []
                    for line in lines:
                        if "GPU[" in line and ":" in line:
                            name = line.split(":", 1)[1].strip()
                            gpu_names.append(name)
                    
                    for i, name in enumerate(gpu_names):
                        gpu_info["devices"].append({
                            "id": i,
                            "name": name,
                            "type": "rocm"
                        })
                    gpu_info["count"] = len(gpu_info["devices"])
            except:
                pass
                
        return gpu_info
    
    def get_startup_script(self, coordinator_url, api_key, worker_id=None):
        """Generate a Linux worker startup script."""
        if not worker_id:
            worker_id = f"worker_{uuid.uuid4().hex[:8]}"
            
        script = f"""#!/bin/bash
# Auto-generated worker startup script

# Set up environment
export WORKER_ID="{worker_id}"
export COORDINATOR_URL="{coordinator_url}"
export API_KEY="{api_key}"

# Create log directory
mkdir -p logs

# Start worker in the background
nohup python run_worker_client.py \\
    --coordinator "$COORDINATOR_URL" \\
    --api-key "$API_KEY" \\
    --worker-id "$WORKER_ID" \\
    --log-file "logs/$WORKER_ID.log" \\
    > "logs/$WORKER_ID.out" 2> "logs/$WORKER_ID.err" &

echo "Worker $WORKER_ID started, connecting to $COORDINATOR_URL"
echo "Worker PID: $!"
echo "Logs available in logs/$WORKER_ID.log"
"""
        return script


class ContainerPlatformHandler(PlatformHandler):
    """Container-specific platform handler."""
    
    def get_worker_command(self, config):
        """Get the command to start a worker in a container."""
        cmd = [
            "python", 
            "run_worker_client.py",
            "--coordinator", config.get("coordinator_url", "http://coordinator:8080"),
            "--api-key", config.get("api_key", "default_key")
        ]
        
        if "worker_id" in config:
            cmd.extend(["--worker-id", config["worker_id"]])
            
        # Container-specific settings
        cmd.extend(["--container-mode"])
        
        if config.get("log_to_file"):
            cmd.extend(["--log-file", "/logs/worker.log"])
            
        return " ".join(cmd)
    
    def create_deployment_script(self, config, output_path):
        """Create a container deployment script (Docker Compose)."""
        compose_content = """version: '3'

services:
  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    environment:
      - COORDINATOR_URL={coordinator_url}
      - API_KEY={api_key}
      - WORKER_ID={worker_id}
      - CONTAINER_ENV=1
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    networks:
      - testing_network

networks:
  testing_network:
    driver: bridge
""".format(
            coordinator_url=config.get("coordinator_url", "http://coordinator:8080"),
            api_key=config.get("api_key", "default_key"),
            worker_id=config.get("worker_id", "worker_" + str(uuid.uuid4())[:8])
        )
        
        # Create docker-compose.yml
        with open(output_path, "w") as f:
            f.write(compose_content)
            
        # Create Dockerfile.worker
        dockerfile_path = os.path.join(os.path.dirname(output_path), "Dockerfile.worker")
        dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker code
COPY . .

# Create log directory
RUN mkdir -p logs

# Entry point
CMD ["python", "run_worker_client.py", "--coordinator", "${COORDINATOR_URL}", "--api-key", "${API_KEY}", "--worker-id", "${WORKER_ID}", "--log-file", "/app/logs/worker.log", "--container-mode"]
"""
        
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
            
        return output_path
```

### Phase 5: CI/CD Pipeline Integration

#### Tasks:
- [ ] Develop GitHub Actions workflow for distributed testing
- [ ] Create coordinator deployment process for CI/CD
- [ ] Implement automatic worker provisioning on PR events
- [ ] Add test status reporting back to GitHub
- [ ] Create comprehensive PR review dashboards

#### Implementation Details:
```yaml
# .github/workflows/distributed-testing.yml
name: Distributed Testing Framework

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:
    inputs:
      test_filter:
        description: 'Test filter to run (e.g., "test_bert", "models")'
        required: false
        default: ''

jobs:
  setup-coordinator:
    runs-on: ubuntu-latest
    outputs:
      coordinator_url: ${{ steps.start-coordinator.outputs.coordinator_url }}
      api_key: ${{ steps.start-coordinator.outputs.api_key }}
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r test/duckdb_api/distributed_testing/requirements.txt
          
      - name: Start coordinator
        id: start-coordinator
        run: |
          # Generate random API key
          API_KEY=$(python -c "import uuid; print(uuid.uuid4().hex)")
          echo "api_key=$API_KEY" >> $GITHUB_OUTPUT
          
          # Start coordinator in background
          nohup python test/duckdb_api/distributed_testing/run_coordinator_server.py \
            --host 0.0.0.0 \
            --port 8080 \
            --api-key $API_KEY \
            --db-path ./test_results.duckdb \
            > coordinator.log 2>&1 &
          
          # Wait for coordinator to start
          sleep 5
          
          # Get public URL (using ngrok in GitHub Actions)
          pip install pyngrok
          python -c "from pyngrok import ngrok; tunnel = ngrok.connect(8080); print(f'coordinator_url={tunnel.public_url}', end='')" >> $GITHUB_OUTPUT
          
          # Verify coordinator is running
          curl localhost:8080/health
  
  start-workers:
    needs: setup-coordinator
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        hardware: ['cpu', 'integration']
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r test/duckdb_api/distributed_testing/requirements.txt
          
      - name: Start worker
        run: |
          # Generate worker ID
          WORKER_ID="${{ matrix.os }}-${{ matrix.hardware }}-${{ github.run_id }}"
          
          # Start worker
          python test/duckdb_api/distributed_testing/run_worker_client.py \
            --coordinator ${{ needs.setup-coordinator.outputs.coordinator_url }} \
            --api-key ${{ needs.setup-coordinator.outputs.api_key }} \
            --worker-id "$WORKER_ID" \
            --worker-type "${{ matrix.hardware }}" \
            --log-level DEBUG \
            > worker.log 2>&1 &
            
          # Wait for worker to register
          sleep 10
          
          # Keep worker running in background
          echo "Worker started with ID: $WORKER_ID"
  
  generate-and-run-tests:
    needs: [setup-coordinator, start-workers]
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r test/duckdb_api/distributed_testing/requirements.txt
          
      - name: Generate and submit tests
        run: |
          # Generate tests
          python test/duckdb_api/distributed_testing/cicd_integration.py \
            --generate-tests \
            --test-dir ./test \
            --output-dir ./generated_tests \
            --filter "${{ github.event.inputs.test_filter || '' }}"
          
          # Submit tests to coordinator
          python test/duckdb_api/distributed_testing/cicd_integration.py \
            --submit-tests \
            --coordinator ${{ needs.setup-coordinator.outputs.coordinator_url }} \
            --api-key ${{ needs.setup-coordinator.outputs.api_key }} \
            --test-dir ./generated_tests
          
          # Wait for tests to complete and collect results
          python test/duckdb_api/distributed_testing/cicd_integration.py \
            --wait-for-results \
            --coordinator ${{ needs.setup-coordinator.outputs.coordinator_url }} \
            --api-key ${{ needs.setup-coordinator.outputs.api_key }} \
            --output-dir ./test_results \
            --timeout 1800
      
      - name: Generate test reports
        run: |
          python test/duckdb_api/distributed_testing/cicd_integration.py \
            --generate-reports \
            --input-dir ./test_results \
            --output-dir ./reports \
            --formats json,md,html
      
      - name: Upload test results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: ./test_results
          
      - name: Upload test reports
        uses: actions/upload-artifact@v3
        with:
          name: test-reports
          path: ./reports
          
      - name: Check test status
        run: |
          if grep -q '"failed": true' ./test_results/summary.json; then
            echo "Tests failed. See reports for details."
            exit 1
          else
            echo "All tests passed!"
          fi
```

### Phase 6: Dynamic Resource Management

#### Tasks:
- [ ] Implement resource tracking with fine-grained metrics
- [ ] Create adaptive scaling based on workload patterns
- [ ] Develop dynamic worker capabilities reassessment
- [ ] Add support for ephemeral workers with cloud integration
- [ ] Implement resource reservation and release tracking

#### Implementation Details:
```python
class DynamicResourceManager:
    """Manages resources dynamically with adaptive scaling."""
    
    def __init__(self, db_manager=None):
        """Initialize the dynamic resource manager."""
        self.db_manager = db_manager
        self.worker_resources = {}  # worker_id -> resource profile
        self.resource_reservations = {}  # (worker_id, resource_type) -> list of reservations
        self.worker_performance = {}  # worker_id -> performance metrics
        self.resource_lock = threading.RLock()
        
        # Scaling parameters
        self.scaling_config = {
            "min_workers": 1,
            "max_workers": 10,
            "target_utilization": 0.7,  # 70% utilization target
            "scale_up_threshold": 0.8,  # Scale up at 80% utilization
            "scale_down_threshold": 0.3,  # Scale down at 30% utilization
            "cooldown_period_seconds": 60,
            "evaluation_window_seconds": 300  # 5 minutes
        }
        
        # Cloud provider integration
        self.cloud_provider = None
        
        # Resources history for tracking
        self.resource_history = {}  # worker_id -> list of (timestamp, metrics)
        self.history_lock = threading.Lock()
        
        # Performance prediction
        self.perf_predictor = ResourcePerformancePredictor()
    
    def register_worker(self, worker_id, capabilities):
        """Register a worker with its capabilities."""
        with self.resource_lock:
            # Initialize resource profile
            self.worker_resources[worker_id] = {
                "cpu": {
                    "total": capabilities.get("cpu", {}).get("count", 0),
                    "available": capabilities.get("cpu", {}).get("count", 0),
                    "reserved": 0
                },
                "memory": {
                    "total_gb": capabilities.get("memory", {}).get("total_gb", 0),
                    "available_gb": capabilities.get("memory", {}).get("total_gb", 0),
                    "reserved_gb": 0
                },
                "gpu": {}
            }
            
            # Initialize GPU resources if available
            gpu_devices = capabilities.get("gpu", {}).get("devices", [])
            for device in gpu_devices:
                device_id = device.get("id", 0)
                device_type = device.get("type", "unknown")
                device_key = f"{device_type}:{device_id}"
                
                self.worker_resources[worker_id]["gpu"][device_key] = {
                    "id": device_id,
                    "type": device_type,
                    "total_memory_gb": device.get("memory_gb", 0),
                    "available_memory_gb": device.get("memory_gb", 0),
                    "reserved_memory_gb": 0,
                    "brand": device.get("brand", "unknown")
                }
            
            # Initialize resource reservations
            self.resource_reservations[worker_id] = {
                "cpu": [],
                "memory": [],
                "gpu": {}
            }
            
            for device_key in self.worker_resources[worker_id]["gpu"]:
                self.resource_reservations[worker_id]["gpu"][device_key] = []
            
            # Initialize performance metrics
            self.worker_performance[worker_id] = {
                "tasks_completed": 0,
                "total_execution_time": 0,
                "avg_cpu_usage": 0,
                "avg_memory_usage": 0,
                "start_time": time.time(),
                "last_active": time.time()
            }
            
            # Initialize resource history
            with self.history_lock:
                self.resource_history[worker_id] = []
                
            return True
    
    def reserve_resources(self, worker_id, task_id, requirements):
        """Reserve resources for a task on a worker."""
        with self.resource_lock:
            if worker_id not in self.worker_resources:
                return False, "Worker not registered"
                
            # Check if required resources are available
            cpu_required = requirements.get("cpu_cores", 1)
            memory_required = requirements.get("memory_gb", 1.0)
            gpu_required = requirements.get("gpu_memory_gb", 0)
            gpu_type = requirements.get("gpu_type")
            
            worker_resources = self.worker_resources[worker_id]
            
            # Check CPU availability
            if worker_resources["cpu"]["available"] < cpu_required:
                return False, f"Not enough CPU cores available: {worker_resources['cpu']['available']} < {cpu_required}"
                
            # Check memory availability
            if worker_resources["memory"]["available_gb"] < memory_required:
                return False, f"Not enough memory available: {worker_resources['memory']['available_gb']} < {memory_required}"
                
            # Check GPU availability if required
            if gpu_required > 0:
                if not gpu_type:
                    # Find any suitable GPU
                    suitable_gpu = None
                    for device_key, device in worker_resources["gpu"].items():
                        if device["available_memory_gb"] >= gpu_required:
                            suitable_gpu = device_key
                            break
                            
                    if not suitable_gpu:
                        return False, f"No GPU with {gpu_required} GB available"
                        
                    gpu_type = suitable_gpu
                elif gpu_type not in worker_resources["gpu"]:
                    return False, f"GPU type {gpu_type} not available"
                elif worker_resources["gpu"][gpu_type]["available_memory_gb"] < gpu_required:
                    return False, f"Not enough GPU memory available: {worker_resources['gpu'][gpu_type]['available_memory_gb']} < {gpu_required}"
            
            # All resources available, reserve them
            worker_resources["cpu"]["available"] -= cpu_required
            worker_resources["cpu"]["reserved"] += cpu_required
            worker_resources["memory"]["available_gb"] -= memory_required
            worker_resources["memory"]["reserved_gb"] += memory_required
            
            # Create reservation records
            cpu_reservation = {
                "task_id": task_id,
                "amount": cpu_required,
                "timestamp": time.time()
            }
            self.resource_reservations[worker_id]["cpu"].append(cpu_reservation)
            
            memory_reservation = {
                "task_id": task_id,
                "amount": memory_required,
                "timestamp": time.time()
            }
            self.resource_reservations[worker_id]["memory"].append(memory_reservation)
            
            # Reserve GPU if required
            gpu_reservation = None
            if gpu_required > 0:
                worker_resources["gpu"][gpu_type]["available_memory_gb"] -= gpu_required
                worker_resources["gpu"][gpu_type]["reserved_memory_gb"] += gpu_required
                
                gpu_reservation = {
                    "task_id": task_id,
                    "amount": gpu_required,
                    "timestamp": time.time(),
                    "device_key": gpu_type
                }
                self.resource_reservations[worker_id]["gpu"][gpu_type].append(gpu_reservation)
            
            # Update worker performance tracking
            self.worker_performance[worker_id]["last_active"] = time.time()
            
            # Record reservation for tracking
            reservation = {
                "task_id": task_id,
                "worker_id": worker_id,
                "cpu_cores": cpu_required,
                "memory_gb": memory_required,
                "gpu_memory_gb": gpu_required,
                "gpu_type": gpu_type if gpu_required > 0 else None,
                "timestamp": time.time()
            }
            
            # Record current resource state in history
            self._record_resource_state(worker_id)
            
            return True, reservation
    
    def release_resources(self, worker_id, task_id):
        """Release resources reserved for a task."""
        with self.resource_lock:
            if worker_id not in self.worker_resources:
                return False, "Worker not registered"
                
            worker_resources = self.worker_resources[worker_id]
            resource_reservations = self.resource_reservations[worker_id]
            
            # Find and release CPU reservation
            cpu_released = 0
            for i, reservation in enumerate(resource_reservations["cpu"]):
                if reservation["task_id"] == task_id:
                    cpu_released = reservation["amount"]
                    worker_resources["cpu"]["available"] += cpu_released
                    worker_resources["cpu"]["reserved"] -= cpu_released
                    resource_reservations["cpu"].pop(i)
                    break
            
            # Find and release memory reservation
            memory_released = 0
            for i, reservation in enumerate(resource_reservations["memory"]):
                if reservation["task_id"] == task_id:
                    memory_released = reservation["amount"]
                    worker_resources["memory"]["available_gb"] += memory_released
                    worker_resources["memory"]["reserved_gb"] -= memory_released
                    resource_reservations["memory"].pop(i)
                    break
            
            # Find and release GPU reservation if any
            gpu_released = 0
            gpu_type = None
            for device_key, reservations in resource_reservations["gpu"].items():
                for i, reservation in enumerate(reservations):
                    if reservation["task_id"] == task_id:
                        gpu_released = reservation["amount"]
                        gpu_type = device_key
                        worker_resources["gpu"][device_key]["available_memory_gb"] += gpu_released
                        worker_resources["gpu"][device_key]["reserved_memory_gb"] -= gpu_released
                        reservations.pop(i)
                        break
                if gpu_type:
                    break
            
            # Update worker performance tracking
            self.worker_performance[worker_id]["last_active"] = time.time()
            self.worker_performance[worker_id]["tasks_completed"] += 1
            
            # Record current resource state in history
            self._record_resource_state(worker_id)
            
            return True, {
                "task_id": task_id,
                "worker_id": worker_id,
                "cpu_cores_released": cpu_released,
                "memory_gb_released": memory_released,
                "gpu_memory_gb_released": gpu_released,
                "gpu_type": gpu_type
            }
    
    def _record_resource_state(self, worker_id):
        """Record the current resource state for historical tracking."""
        with self.history_lock:
            if worker_id not in self.resource_history:
                self.resource_history[worker_id] = []
                
            worker_resources = copy.deepcopy(self.worker_resources[worker_id])
            timestamp = time.time()
            
            # Calculate utilization percentages
            if worker_resources["cpu"]["total"] > 0:
                cpu_utilization = worker_resources["cpu"]["reserved"] / worker_resources["cpu"]["total"]
            else:
                cpu_utilization = 0
                
            if worker_resources["memory"]["total_gb"] > 0:
                memory_utilization = worker_resources["memory"]["reserved_gb"] / worker_resources["memory"]["total_gb"]
            else:
                memory_utilization = 0
                
            gpu_utilization = {}
            for device_key, device in worker_resources["gpu"].items():
                if device["total_memory_gb"] > 0:
                    gpu_utilization[device_key] = device["reserved_memory_gb"] / device["total_memory_gb"]
                else:
                    gpu_utilization[device_key] = 0
            
            # Record state
            state = {
                "timestamp": timestamp,
                "cpu_utilization": cpu_utilization,
                "memory_utilization": memory_utilization,
                "gpu_utilization": gpu_utilization,
                "cpu_available": worker_resources["cpu"]["available"],
                "memory_available_gb": worker_resources["memory"]["available_gb"],
                "cpu_reserved": worker_resources["cpu"]["reserved"],
                "memory_reserved_gb": worker_resources["memory"]["reserved_gb"]
            }
            
            self.resource_history[worker_id].append(state)
            
            # Trim history if too large (keep last 1000 entries)
            if len(self.resource_history[worker_id]) > 1000:
                self.resource_history[worker_id] = self.resource_history[worker_id][-1000:]
    
    def get_worker_utilization(self, worker_id):
        """Get current utilization metrics for a worker."""
        with self.resource_lock:
            if worker_id not in self.worker_resources:
                return None
                
            worker_resources = self.worker_resources[worker_id]
            
            # Calculate utilization percentages
            if worker_resources["cpu"]["total"] > 0:
                cpu_utilization = worker_resources["cpu"]["reserved"] / worker_resources["cpu"]["total"]
            else:
                cpu_utilization = 0
                
            if worker_resources["memory"]["total_gb"] > 0:
                memory_utilization = worker_resources["memory"]["reserved_gb"] / worker_resources["memory"]["total_gb"]
            else:
                memory_utilization = 0
                
            gpu_utilization = {}
            for device_key, device in worker_resources["gpu"].items():
                if device["total_memory_gb"] > 0:
                    gpu_utilization[device_key] = device["reserved_memory_gb"] / device["total_memory_gb"]
                else:
                    gpu_utilization[device_key] = 0
            
            # Calculate overall utilization as weighted average
            overall_utilization = (
                0.4 * cpu_utilization + 
                0.4 * memory_utilization + 
                0.2 * (sum(gpu_utilization.values()) / max(1, len(gpu_utilization)))
            )
            
            return {
                "worker_id": worker_id,
                "cpu_utilization": cpu_utilization,
                "memory_utilization": memory_utilization,
                "gpu_utilization": gpu_utilization,
                "overall_utilization": overall_utilization,
                "last_active": self.worker_performance[worker_id]["last_active"],
                "tasks_completed": self.worker_performance[worker_id]["tasks_completed"]
            }
    
    def evaluate_scaling(self):
        """Evaluate if the system should scale up or down based on utilization."""
        # Get overall system utilization
        total_utilization = 0
        total_workers = 0
        
        for worker_id in self.worker_resources:
            utilization = self.get_worker_utilization(worker_id)
            if utilization:
                total_utilization += utilization["overall_utilization"]
                total_workers += 1
        
        if total_workers == 0:
            return "no_change", 0
            
        avg_utilization = total_utilization / total_workers
        
        # Check if we need to scale up or down
        if avg_utilization > self.scaling_config["scale_up_threshold"]:
            # Check if we're below max workers
            if total_workers < self.scaling_config["max_workers"]:
                # Scale up
                workers_to_add = min(
                    self.scaling_config["max_workers"] - total_workers,
                    max(1, int(total_workers * 0.2))  # Add at least 1, at most 20% of current
                )
                return "scale_up", workers_to_add
        elif avg_utilization < self.scaling_config["scale_down_threshold"]:
            # Check if we're above min workers
            if total_workers > self.scaling_config["min_workers"]:
                # Scale down
                workers_to_remove = min(
                    total_workers - self.scaling_config["min_workers"],
                    max(1, int(total_workers * 0.2))  # Remove at least 1, at most 20% of current
                )
                return "scale_down", workers_to_remove
        
        return "no_change", 0
    
    def get_worker_to_scale_down(self):
        """Get the best worker to remove when scaling down."""
        candidates = []
        
        with self.resource_lock:
            for worker_id, resources in self.worker_resources.items():
                # Skip workers with active tasks
                if resources["cpu"]["reserved"] > 0 or resources["memory"]["reserved_gb"] > 0:
                    continue
                    
                gpu_reserved = False
                for device in resources["gpu"].values():
                    if device["reserved_memory_gb"] > 0:
                        gpu_reserved = True
                        break
                
                if gpu_reserved:
                    continue
                    
                # This worker has no reserved resources
                utilization = self.get_worker_utilization(worker_id)
                last_active = utilization["last_active"]
                candidates.append((worker_id, last_active))
        
        if not candidates:
            return None
            
        # Get the least recently active worker
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    
    def predict_resource_needs(self, task_type, task_requirements):
        """Predict resource requirements for a task based on historical data."""
        return self.perf_predictor.predict_resource_requirements(task_type, task_requirements)


class ResourcePerformancePredictor:
    """Predicts resource requirements based on historical task execution data."""
    
    def __init__(self):
        """Initialize the resource performance predictor."""
        self.task_history = {}  # (task_type, model_type) -> list of execution records
        self.lock = threading.Lock()
    
    def record_task_execution(self, task_type, model_type, batch_size, actual_resources, execution_time):
        """Record the resources used and execution time for a task."""
        with self.lock:
            key = (task_type, model_type)
            if key not in self.task_history:
                self.task_history[key] = []
                
            record = {
                "batch_size": batch_size,
                "resources": actual_resources,
                "execution_time": execution_time,
                "timestamp": time.time()
            }
            
            self.task_history[key].append(record)
            
            # Trim history if too large (keep last 100 entries)
            if len(self.task_history[key]) > 100:
                self.task_history[key] = self.task_history[key][-100:]
    
    def predict_resource_requirements(self, task_type, task_requirements):
        """Predict resource requirements for a task based on historical data."""
        with self.lock:
            model_type = task_requirements.get("model_type", "unknown")
            batch_size = task_requirements.get("batch_size", 1)
            
            key = (task_type, model_type)
            if key not in self.task_history or len(self.task_history[key]) < 5:
                # Not enough history, use provided requirements
                return {
                    "cpu_cores": task_requirements.get("cpu_cores", 1),
                    "memory_gb": task_requirements.get("memory_gb", 1.0),
                    "gpu_memory_gb": task_requirements.get("gpu_memory_gb", 0),
                    "estimated_execution_time": task_requirements.get("estimated_execution_time", 60)
                }
            
            # Find records with similar batch size
            similar_records = []
            for record in self.task_history[key]:
                if abs(record["batch_size"] - batch_size) <= 2:  # Allow batch sizes within 2
                    similar_records.append(record)
            
            if not similar_records:
                # No similar batch sizes, use all records
                similar_records = self.task_history[key]
            
            # Calculate averages
            avg_cpu = sum(r["resources"].get("cpu_cores", 1) for r in similar_records) / len(similar_records)
            avg_memory = sum(r["resources"].get("memory_gb", 1.0) for r in similar_records) / len(similar_records)
            avg_gpu = sum(r["resources"].get("gpu_memory_gb", 0) for r in similar_records) / len(similar_records)
            avg_time = sum(r["execution_time"] for r in similar_records) / len(similar_records)
            
            # Apply a small buffer (10%) for safety
            cpu_estimate = avg_cpu * 1.1
            memory_estimate = avg_memory * 1.1
            gpu_estimate = avg_gpu * 1.1 if avg_gpu > 0 else 0
            time_estimate = avg_time * 1.1
            
            return {
                "cpu_cores": max(1, int(cpu_estimate)),
                "memory_gb": max(0.5, memory_estimate),
                "gpu_memory_gb": max(0, gpu_estimate),
                "estimated_execution_time": max(10, time_estimate)
            }
    
    def get_resource_scaling_factor(self, task_type, model_type, batch_size_ratio):
        """Estimate how resources scale with batch size for a specific task and model type."""
        with self.lock:
            key = (task_type, model_type)
            if key not in self.task_history or len(self.task_history[key]) < 10:
                # Not enough history, use conservative estimates
                return {
                    "cpu_scaling": batch_size_ratio,
                    "memory_scaling": batch_size_ratio,
                    "gpu_scaling": batch_size_ratio,
                    "time_scaling": batch_size_ratio**0.8  # Time often scales sublinearly
                }
            
            # Group by batch size
            batch_groups = {}
            for record in self.task_history[key]:
                bs = record["batch_size"]
                if bs not in batch_groups:
                    batch_groups[bs] = []
                batch_groups[bs].append(record)
            
            # Need at least 2 batch sizes to estimate scaling
            if len(batch_groups) < 2:
                return {
                    "cpu_scaling": batch_size_ratio,
                    "memory_scaling": batch_size_ratio,
                    "gpu_scaling": batch_size_ratio,
                    "time_scaling": batch_size_ratio**0.8
                }
            
            # Calculate average resources for each batch size
            batch_averages = {}
            for bs, records in batch_groups.items():
                batch_averages[bs] = {
                    "cpu": sum(r["resources"].get("cpu_cores", 1) for r in records) / len(records),
                    "memory": sum(r["resources"].get("memory_gb", 1.0) for r in records) / len(records),
                    "gpu": sum(r["resources"].get("gpu_memory_gb", 0) for r in records) / len(records),
                    "time": sum(r["execution_time"] for r in records) / len(records)
                }
            
            # Sort batch sizes
            batch_sizes = sorted(batch_averages.keys())
            
            # Calculate scaling factors between consecutive batch sizes
            scaling_factors = {
                "cpu": [],
                "memory": [],
                "gpu": [],
                "time": []
            }
            
            for i in range(1, len(batch_sizes)):
                bs1 = batch_sizes[i-1]
                bs2 = batch_sizes[i]
                ratio = bs2 / bs1
                
                for resource in ["cpu", "memory", "gpu", "time"]:
                    if batch_averages[bs1][resource] > 0:
                        factor = batch_averages[bs2][resource] / batch_averages[bs1][resource]
                        # Normalize by batch size ratio
                        scaling_factors[resource].append(factor / ratio)
            
            # Average scaling factors
            avg_scaling = {
                "cpu_scaling": sum(scaling_factors["cpu"]) / len(scaling_factors["cpu"]) if scaling_factors["cpu"] else 1.0,
                "memory_scaling": sum(scaling_factors["memory"]) / len(scaling_factors["memory"]) if scaling_factors["memory"] else 1.0,
                "gpu_scaling": sum(scaling_factors["gpu"]) / len(scaling_factors["gpu"]) if scaling_factors["gpu"] else 1.0,
                "time_scaling": sum(scaling_factors["time"]) / len(scaling_factors["time"]) if scaling_factors["time"] else 0.8
            }
            
            # Apply scaling to the requested ratio
            return {
                "cpu_scaling": batch_size_ratio ** avg_scaling["cpu_scaling"],
                "memory_scaling": batch_size_ratio ** avg_scaling["memory_scaling"],
                "gpu_scaling": batch_size_ratio ** avg_scaling["gpu_scaling"],
                "time_scaling": batch_size_ratio ** avg_scaling["time_scaling"]
            }
```

## Implementation Progress

### Completed Phases
- ✅ **Phase 1: DuckDB Integration Enhancement** - Completed March 15, 2025
  - Implemented DuckDBResultProcessor for direct database operations
  - Created WorkerDuckDBIntegration for worker-side result management
  - Added CoordinatorDuckDBIntegration for centralized result processing
  - Implemented comprehensive test suite for validation

- ✅ **Phase 2: Template-Based Test Generator Integration** - Completed March 16, 2025
  - Created TestGeneratorIntegration for dynamic test generation
  - Implemented template database for storing and retrieving templates
  - Added model family detection and mapping
  - Developed test dependency tracking for proper execution order
  - Added resource estimation for memory and priority calculation

- ✅ **Phase 4: Cross-Platform Worker Support** - Completed March 20, 2025
  - Created CrossPlatformWorkerSupport with platform-specific handlers
  - Implemented unified interface for platform-specific operations
  - Added comprehensive hardware detection for each platform
  - Developed platform-specific deployment script generation
  - Created path conversion utilities for cross-platform compatibility
  - Added container support for Docker and Kubernetes environments
  - Implemented unit tests and example scripts

### Remaining Phases
- ⏸️ **Phase 3: Enhanced JWT Authentication** - Deferred
- 📅 **Phase 5: CI/CD Pipeline Integration** - Not Started
- 📅 **Phase 6: Dynamic Resource Management** - Not Started

## Expected Outcome

Upon implementation of these enhancements, the Distributed Testing Framework will provide:

1. ✅ Seamless integration with the existing DuckDB database system for efficient result storage and retrieval
2. ✅ Automated test generation and execution across heterogeneous hardware environments
3. ⏸️ Secure communication between coordinators and workers with robust authentication (Deferred)
4. ✅ Cross-platform compatibility across Linux, macOS, Windows, and container environments
5. 📅 Full CI/CD integration for automated testing during development workflows
6. 📅 Intelligent resource management with adaptive scaling based on workload patterns

## Testing Plan

To ensure successful implementation, we will develop a comprehensive testing methodology:

1. **Unit Tests**: Verify individual components function correctly
2. **Integration Tests**: Validate interaction between components
3. **End-to-End Tests**: Test complete workflows from test generation to result storage
4. **Fault Tolerance Tests**: Verify system resilience during failures
5. **Load Tests**: Evaluate performance under high concurrency
6. **Cross-Platform Tests**: Verify functionality across different environments

## Timeline

- ✅ Week 1: DuckDB Integration Enhancement (Completed March 15, 2025)
- ✅ Week 2: Template-Based Test Generator Integration (Completed March 16, 2025)
- ✅ Week 3: Cross-Platform Worker Support (Completed March 20, 2025)
- 📅 Week 4: CI/CD Pipeline Integration (Target: March 29, 2025)
- 📅 Week 5: Dynamic Resource Management (Target: April 5, 2025)
- 📅 Week 6-7: Testing, Optimization, and Documentation (Target: April 19, 2025)
- ⏸️ Enhanced JWT Authentication (Deferred for future implementation)

## Conclusion

This integration plan will elevate the Distributed Testing Framework from a collection of connected components to a fully integrated system that effectively leverages distributed resources for efficient testing. The implementation will prioritize security, reliability, and adaptability to various testing scenarios and hardware environments.