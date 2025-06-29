# Distributed Testing Framework - Deployment Guide

This guide provides comprehensive instructions for deploying the IPFS Accelerate Distributed Testing Framework in a production environment.

## Architecture Overview

The distributed testing framework consists of the following components:

1. **Coordinator**: Central server that manages workers and schedules tasks
2. **Workers**: Distributed nodes that execute tests on various hardware
3. **Dashboard**: Web interface for monitoring the system
4. **Task Submission Client**: Tool for submitting tasks to the coordinator

![Architecture Diagram](https://mermaid.ink/img/pako:eNqFVLGO2zAM_RWiQzpU9pAmaFd3qJMg6GX10DlyLMaWDEmOm8PB_16KdhznaOgWPpFPJB95qG-miiSN-ULLLfKb_Ja8d-jIqvqPVYW-dOR7_vSlDh1txaMpnKu1ZMGU38qWFmq3HcixMYJCgvYXLYvTTcsmJxeIv5UD2mClJkc3pVZQEPdoCK4K5dAJRm4Cg22t5I06OlCE9UBeGSdqqTXhXrjq5ELI3Fq5uHtv5d9Ic1GUmv5PBvZXfqnk7WQKV8Hf4qqoQ7XAuZI9WyYXqHYK8o5Q2yW51qxHVXXcwICedPHXMWZNS94kQ0c9Kt6SZcibVaJ9p5qJ7UlAyHvUb9V-1lTGDdgD4Dyk14dRlKqCkQNnL_A1aHh6q7HtWaNsw1QmPZR8JxzJuCYvGu1a_nj_fHwLtHiXf13_c8cKb_B1HEaIGTsHcjyePcLgqk7mdR_P4-N-2f6h-P0L9LVKU9LhkKZpwsLjvxBGcq3IHyieR9mh9T2yN5XJ5pQPtjRVZFwcYyepJ3lXLPRG4TpkIBfHLXc94vwcTz29aG6aJBCwDz5uoSB0X2I9M_XmmkL_w9RZRG2j_SbZYz7gLv4_OEW6KQ?type=png)

## Prerequisites

### System Requirements

- **Coordinator Server**:
  - 4+ CPU cores
  - 8+ GB RAM
  - 100+ GB storage
  - Ubuntu 20.04 LTS or later

- **Worker Nodes**:
  - Hardware appropriate for testing (CPUs, GPUs, etc.)
  - 4+ GB RAM per node
  - 50+ GB storage per node
  - Ubuntu 20.04 LTS or compatible OS

- **Dashboard Server**:
  - 2+ CPU cores
  - 4+ GB RAM
  - Ubuntu 20.04 LTS or later

### Software Requirements

- Python 3.8 or later
- Docker (optional, for containerized deployment)
- PostgreSQL (for production coordinator)
- Redis (optional, for improved performance)
- Nginx (for TLS termination and load balancing)

## Deployment Options

The framework can be deployed in several ways:

1. **Bare Metal**: Direct installation on physical servers
2. **Virtual Machines**: Deployment on VMs (AWS EC2, GCP VMs, etc.)
3. **Docker**: Containerized deployment using Docker
4. **Kubernetes**: Orchestrated deployment with Kubernetes
5. **Hybrid**: Coordinator in cloud, workers on various hardware

This guide focuses on the Docker deployment option, which provides the best balance of flexibility and ease of setup.

## Coordinator Deployment

### Docker Deployment

1. Create a configuration directory:

```bash
mkdir -p /opt/distributed-testing/config
mkdir -p /opt/distributed-testing/data
```

2. Create a coordinator configuration file:

```bash
cat > /opt/distributed-testing/config/coordinator.json << 'EOF'
{
  "host": "0.0.0.0",
  "port": 8080,
  "database": {
    "type": "postgresql",
    "host": "db",
    "port": 5432,
    "name": "distributed_testing",
    "user": "dtuser",
    "password": "CHANGE_ME_TO_SECURE_PASSWORD"
  },
  "redis": {
    "host": "redis",
    "port": 6379
  },
  "logging": {
    "level": "info",
    "format": "json",
    "path": "/var/log/distributed-testing"
  },
  "task_queue": {
    "batch_size": 10,
    "max_retries": 3
  },
  "worker_management": {
    "heartbeat_interval": 10,
    "worker_timeout": 30,
    "auto_recovery": true
  },
  "high_availability": {
    "enabled": true,
    "consensus_protocol": "raft"
  },
  "security": {
    "api_keys_enabled": true,
    "token_expiry": 3600
  }
}
EOF
```

3. Create a `docker-compose.yml` file:

```bash
cat > /opt/distributed-testing/docker-compose.yml << 'EOF'
version: '3.8'

services:
  coordinator:
    image: ipfs-accelerate/distributed-testing-coordinator:latest
    ports:
      - "8080:8080"
    volumes:
      - ./config:/etc/distributed-testing
      - ./data:/var/lib/distributed-testing
    environment:
      - CONFIG_FILE=/etc/distributed-testing/coordinator.json
      - DB_PASSWORD=CHANGE_ME_TO_SECURE_PASSWORD
    depends_on:
      - db
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/status"]
      interval: 1m
      timeout: 10s
      retries: 3

  dashboard:
    image: ipfs-accelerate/distributed-testing-dashboard:latest
    ports:
      - "8050:8050"
    environment:
      - COORDINATOR_URL=http://coordinator:8080
    depends_on:
      - coordinator
    restart: unless-stopped

  db:
    image: postgres:14
    volumes:
      - pg_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=dtuser
      - POSTGRES_PASSWORD=CHANGE_ME_TO_SECURE_PASSWORD
      - POSTGRES_DB=distributed_testing
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "dtuser"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:6
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  pg_data:
  redis_data:
EOF
```

4. Start the coordinator and related services:

```bash
cd /opt/distributed-testing
docker-compose up -d
```

5. Create API key for worker authentication:

```bash
docker-compose exec coordinator python -m distributed_testing.create_api_key --name "worker-key" --role "worker"
```

Save the generated API key for worker deployment.

### Bare Metal Deployment

For bare metal deployments, follow these steps:

1. Install dependencies:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv postgresql redis-server
```

2. Create a Python virtual environment:

```bash
python3 -m venv /opt/distributed-testing/venv
source /opt/distributed-testing/venv/bin/activate
```

3. Install the distributed testing package:

```bash
pip install ipfs-accelerate-distributed-testing
```

4. Configure PostgreSQL:

```bash
sudo -u postgres psql -c "CREATE USER dtuser WITH PASSWORD 'CHANGE_ME_TO_SECURE_PASSWORD';"
sudo -u postgres psql -c "CREATE DATABASE distributed_testing OWNER dtuser;"
```

5. Create a coordinator configuration file:

```bash
mkdir -p /opt/distributed-testing/config
cat > /opt/distributed-testing/config/coordinator.json << 'EOF'
{
  "host": "0.0.0.0",
  "port": 8080,
  "database": {
    "type": "postgresql",
    "host": "localhost",
    "port": 5432,
    "name": "distributed_testing",
    "user": "dtuser",
    "password": "CHANGE_ME_TO_SECURE_PASSWORD"
  },
  "redis": {
    "host": "localhost",
    "port": 6379
  },
  "logging": {
    "level": "info",
    "format": "json",
    "path": "/var/log/distributed-testing"
  },
  "task_queue": {
    "batch_size": 10,
    "max_retries": 3
  },
  "worker_management": {
    "heartbeat_interval": 10,
    "worker_timeout": 30,
    "auto_recovery": true
  },
  "high_availability": {
    "enabled": false
  },
  "security": {
    "api_keys_enabled": true,
    "token_expiry": 3600
  }
}
EOF
```

6. Create a systemd service file for the coordinator:

```bash
cat > /etc/systemd/system/distributed-testing-coordinator.service << 'EOF'
[Unit]
Description=Distributed Testing Coordinator
After=network.target postgresql.service redis-server.service

[Service]
User=dtuser
Group=dtuser
WorkingDirectory=/opt/distributed-testing
ExecStart=/opt/distributed-testing/venv/bin/python -m distributed_testing.coordinator --config /opt/distributed-testing/config/coordinator.json
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF
```

7. Create a systemd service file for the dashboard:

```bash
cat > /etc/systemd/system/distributed-testing-dashboard.service << 'EOF'
[Unit]
Description=Distributed Testing Dashboard
After=distributed-testing-coordinator.service

[Service]
User=dtuser
Group=dtuser
WorkingDirectory=/opt/distributed-testing
ExecStart=/opt/distributed-testing/venv/bin/python -m distributed_testing.dashboard_server --coordinator http://localhost:8080 --port 8050
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF
```

8. Start the services:

```bash
sudo systemctl daemon-reload
sudo systemctl enable distributed-testing-coordinator
sudo systemctl enable distributed-testing-dashboard
sudo systemctl start distributed-testing-coordinator
sudo systemctl start distributed-testing-dashboard
```

9. Create API key for worker authentication:

```bash
source /opt/distributed-testing/venv/bin/activate
python -m distributed_testing.create_api_key --name "worker-key" --role "worker"
```

Save the generated API key for worker deployment.

## Worker Deployment

Workers should be deployed on machines with the hardware you want to test on. Each worker can have different hardware capabilities.

### Docker Deployment

1. Create a worker configuration directory:

```bash
mkdir -p /opt/distributed-testing-worker/config
```

2. Create a worker configuration file:

```bash
cat > /opt/distributed-testing-worker/config/worker.json << 'EOF'
{
  "coordinator_url": "http://coordinator-hostname:8080",
  "api_key": "YOUR_API_KEY_HERE",
  "heartbeat_interval": 10,
  "reconnect_attempts": 10,
  "db_path": "/var/lib/distributed-testing-worker/worker.duckdb",
  "log_path": "/var/log/distributed-testing-worker",
  "capabilities": {
    "additional_tags": []
  },
  "task_execution": {
    "max_concurrent_tasks": 4
  }
}
EOF
```

Replace `YOUR_API_KEY_HERE` with the API key generated on the coordinator.

3. Create a Docker Compose file:

```bash
cat > /opt/distributed-testing-worker/docker-compose.yml << 'EOF'
version: '3.8'

services:
  worker:
    image: ipfs-accelerate/distributed-testing-worker:latest
    volumes:
      - ./config:/etc/distributed-testing-worker
      - ./data:/var/lib/distributed-testing-worker
    environment:
      - CONFIG_FILE=/etc/distributed-testing-worker/worker.json
    restart: unless-stopped
    # For GPU support with NVIDIA, uncomment the following lines:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]
EOF
```

4. Start the worker:

```bash
cd /opt/distributed-testing-worker
docker-compose up -d
```

### Bare Metal Deployment

For bare metal deployments, follow these steps:

1. Install dependencies:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
```

2. Create a Python virtual environment:

```bash
python3 -m venv /opt/distributed-testing-worker/venv
source /opt/distributed-testing-worker/venv/bin/activate
```

3. Install the distributed testing package:

```bash
pip install ipfs-accelerate-distributed-testing
```

4. Create a worker configuration file:

```bash
mkdir -p /opt/distributed-testing-worker/config
cat > /opt/distributed-testing-worker/config/worker.json << 'EOF'
{
  "coordinator_url": "http://coordinator-hostname:8080",
  "api_key": "YOUR_API_KEY_HERE",
  "heartbeat_interval": 10,
  "reconnect_attempts": 10,
  "db_path": "/var/lib/distributed-testing-worker/worker.duckdb",
  "log_path": "/var/log/distributed-testing-worker",
  "capabilities": {
    "additional_tags": []
  },
  "task_execution": {
    "max_concurrent_tasks": 4
  }
}
EOF
```

Replace `YOUR_API_KEY_HERE` with the API key generated on the coordinator.

5. Create a systemd service file for the worker:

```bash
cat > /etc/systemd/system/distributed-testing-worker.service << 'EOF'
[Unit]
Description=Distributed Testing Worker
After=network.target

[Service]
User=dtuser
Group=dtuser
WorkingDirectory=/opt/distributed-testing-worker
ExecStart=/opt/distributed-testing-worker/venv/bin/python -m distributed_testing.worker --config /opt/distributed-testing-worker/config/worker.json
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF
```

6. Start the worker service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable distributed-testing-worker
sudo systemctl start distributed-testing-worker
```

### Running Multiple Workers

To run multiple workers on the same machine:

1. Create different configuration files for each worker:

```bash
for i in {1..4}; do
  mkdir -p /opt/distributed-testing-worker-$i/config
  cat > /opt/distributed-testing-worker-$i/config/worker.json << EOF
{
  "coordinator_url": "http://coordinator-hostname:8080",
  "api_key": "YOUR_API_KEY_HERE",
  "heartbeat_interval": 10,
  "reconnect_attempts": 10,
  "hostname": "worker-$i",
  "db_path": "/var/lib/distributed-testing-worker-$i/worker.duckdb",
  "log_path": "/var/log/distributed-testing-worker-$i",
  "capabilities": {
    "additional_tags": []
  },
  "task_execution": {
    "max_concurrent_tasks": 1
  }
}
EOF
done
```

2. Create systemd service files for each worker:

```bash
for i in {1..4}; do
  cat > /etc/systemd/system/distributed-testing-worker-$i.service << EOF
[Unit]
Description=Distributed Testing Worker $i
After=network.target

[Service]
User=dtuser
Group=dtuser
WorkingDirectory=/opt/distributed-testing-worker-$i
ExecStart=/opt/distributed-testing-worker/venv/bin/python -m distributed_testing.worker --config /opt/distributed-testing-worker-$i/config/worker.json
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF
done
```

3. Start all worker services:

```bash
sudo systemctl daemon-reload
for i in {1..4}; do
  sudo systemctl enable distributed-testing-worker-$i
  sudo systemctl start distributed-testing-worker-$i
done
```

## Submitting Tasks

You can submit tasks to the coordinator using the submit_tasks.py script:

### Single Task Submission

```bash
python submit_tasks.py --coordinator http://coordinator-hostname:8080 --api-key YOUR_API_KEY_HERE \
  --generate benchmark --model bert-base-uncased --batch-sizes 1,2,4,8,16 --precision fp16,fp32
```

### Batch Task Submission

```bash
python submit_tasks.py --coordinator http://coordinator-hostname:8080 --api-key YOUR_API_KEY_HERE \
  --task-file task_examples.json
```

### Periodic Task Submission

```bash
python submit_tasks.py --coordinator http://coordinator-hostname:8080 --api-key YOUR_API_KEY_HERE \
  --task-file task_examples.json --periodic 3600
```

## Security Considerations

### Network Security

1. **TLS Encryption**: Configure Nginx as a reverse proxy with TLS:

```
server {
    listen 443 ssl;
    server_name coordinator.example.com;

    ssl_certificate /etc/letsencrypt/live/coordinator.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/coordinator.example.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

2. **Firewall Configuration**: Only expose necessary ports:

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 443/tcp
sudo ufw enable
```

### Authentication

1. **API Key Rotation**: Regularly rotate API keys:

```bash
# Generate new key
NEW_KEY=$(docker-compose exec coordinator python -m distributed_testing.create_api_key --name "worker-key-new" --role "worker")

# Update workers with new key
# Then delete old key
docker-compose exec coordinator python -m distributed_testing.delete_api_key --name "worker-key"
```

2. **Dashboard Authentication**: Add basic authentication to the dashboard:

```
location / {
    auth_basic "Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://localhost:8050;
    # ... other proxy settings
}
```

## Monitoring and Maintenance

### Log Monitoring

Configure log rotation:

```bash
cat > /etc/logrotate.d/distributed-testing << 'EOF'
/var/log/distributed-testing/*.log {
  daily
  rotate 14
  compress
  delaycompress
  missingok
  notifempty
  create 0640 dtuser dtuser
}
EOF
```

### Database Backup

Set up regular PostgreSQL backups:

```bash
mkdir -p /opt/distributed-testing/backups

cat > /etc/cron.daily/distributed-testing-backup << 'EOF'
#!/bin/bash
BACKUP_DIR=/opt/distributed-testing/backups
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
pg_dump -U dtuser -d distributed_testing -h localhost -F c -f "$BACKUP_DIR/distributed_testing_$TIMESTAMP.backup"
find $BACKUP_DIR -type f -name "distributed_testing_*.backup" -mtime +14 -delete
EOF

chmod +x /etc/cron.daily/distributed-testing-backup
```

### Health Checks

Set up monitoring for the services:

```bash
# Check coordinator health
curl -f http://localhost:8080/status

# Check worker connectivity
docker-compose exec coordinator python -m distributed_testing.list_workers --status
```

## Scaling the System

### Horizontal Scaling

1. **Coordinator Cluster**: Deploy multiple coordinator instances behind a load balancer
2. **Worker Scaling**: Add more worker nodes as needed
3. **Database Scaling**: Consider database replication for high availability

### Vertical Scaling

1. **Coordinator Resources**: Increase CPU and RAM for the coordinator server
2. **Worker Resources**: Increase CPU and RAM for worker nodes
3. **Database Resources**: Optimize PostgreSQL for better performance

## Troubleshooting

### Common Issues

1. **Coordinator Not Starting**:
   - Check database connectivity
   - Verify configuration file syntax
   - Check system resources

2. **Worker Connection Issues**:
   - Verify network connectivity
   - Check API key validity
   - Inspect worker logs

3. **Task Execution Failures**:
   - Check worker capabilities
   - Verify task parameters
   - Inspect task execution logs

### Logs

- Coordinator logs: `/var/log/distributed-testing/coordinator.log`
- Worker logs: `/var/log/distributed-testing-worker/worker.log`
- Dashboard logs: `/var/log/distributed-testing/dashboard.log`

## Conclusion

This deployment guide provides a comprehensive approach to setting up the IPFS Accelerate Distributed Testing Framework in a production environment. By following these instructions, you can create a robust, scalable, and secure testing infrastructure.