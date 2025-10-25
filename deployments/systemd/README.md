# Systemd Services for IPFS Accelerate

This directory contains optional systemd unit files to run the MCP API and the Performance Dashboard on boot.

## Install

```bash
# Copy units
sudo cp deployments/systemd/ipfs-accelerate-mcp.service /etc/systemd/system/
sudo cp deployments/systemd/ipfs-accelerate-dashboard.service /etc/systemd/system/

# Reload systemd and enable services
sudo systemctl daemon-reload
sudo systemctl enable ipfs-accelerate-mcp.service
sudo systemctl enable ipfs-accelerate-dashboard.service

# Start them now
sudo systemctl start ipfs-accelerate-mcp.service
sudo systemctl start ipfs-accelerate-dashboard.service

# Check status
systemctl status ipfs-accelerate-mcp.service
systemctl status ipfs-accelerate-dashboard.service
```

## Configuration
- Units default to:
  - MCP API on port 8000, binds `0.0.0.0`, CORS `*`.
  - Dashboard on port 8080, binds `0.0.0.0`.
- Adjust environment or ports by editing the unit files or the helper scripts in `tools/`.

## Logs
- MCP API logs: `mcp_server.out`
- Dashboard logs: `dashboard.out`

## Firewall
```bash
sudo ufw allow 8000/tcp
sudo ufw allow 8080/tcp
```