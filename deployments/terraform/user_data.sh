#!/bin/bash
# User data script for AWS EC2 instances

set -euo pipefail

# Configuration from Terraform
ENVIRONMENT="${environment}"

# Update system
apt-get update -y
apt-get upgrade -y

# Install dependencies
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    docker.io \
    git \
    curl \
    wget \
    htop \
    unzip

# Start and enable Docker
systemctl start docker
systemctl enable docker

# Add ubuntu user to docker group
usermod -aG docker ubuntu

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create application directory
mkdir -p /opt/ipfs-accelerate
cd /opt/ipfs-accelerate

# Install IPFS Accelerate Python
python3 -m pip install --upgrade pip
python3 -m pip install ipfs-accelerate-py

# Create systemd service
cat > /etc/systemd/system/ipfs-accelerate.service << EOF
[Unit]
Description=IPFS Accelerate Python ML Platform
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/ipfs-accelerate
Environment=ENVIRONMENT=$ENVIRONMENT
Environment=PORT=8080
ExecStart=/usr/bin/python3 -m ipfs_accelerate_py.main --production --port 8080
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable ipfs-accelerate
systemctl start ipfs-accelerate

# Setup log rotation
cat > /etc/logrotate.d/ipfs-accelerate << EOF
/var/log/ipfs-accelerate/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 ubuntu ubuntu
    postrotate
        systemctl reload ipfs-accelerate
    endscript
}
EOF

# Create health check script
cat > /opt/ipfs-accelerate/health_check.sh << 'EOF'
#!/bin/bash
curl -f http://localhost:8080/health || exit 1
EOF
chmod +x /opt/ipfs-accelerate/health_check.sh

# Install CloudWatch agent (optional)
if [[ "$ENVIRONMENT" == "production" ]]; then
    wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
    dpkg -i amazon-cloudwatch-agent.deb
    
    # CloudWatch agent configuration
    cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << EOF
{
    "metrics": {
        "namespace": "IPFS/Accelerate",
        "metrics_collected": {
            "cpu": {
                "measurement": [
                    "cpu_usage_idle",
                    "cpu_usage_iowait",
                    "cpu_usage_user",
                    "cpu_usage_system"
                ],
                "metrics_collection_interval": 60
            },
            "disk": {
                "measurement": [
                    "used_percent"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "mem": {
                "measurement": [
                    "mem_used_percent"
                ],
                "metrics_collection_interval": 60
            }
        }
    },
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/ipfs-accelerate/app.log",
                        "log_group_name": "/aws/ec2/ipfs-accelerate",
                        "log_stream_name": "{instance_id}/application"
                    }
                ]
            }
        }
    }
}
EOF

    systemctl enable amazon-cloudwatch-agent
    systemctl start amazon-cloudwatch-agent
fi

echo "IPFS Accelerate Python setup completed successfully!"