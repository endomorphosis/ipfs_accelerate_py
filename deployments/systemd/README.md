# Systemd Services for IPFS Accelerate

This directory contains optional systemd unit files to run the MCP API and the Performance Dashboard on boot.

## Install

```bash
# Copy units
sudo cp deployments/systemd/ipfs-accelerate.service /etc/systemd/system/
sudo cp deployments/systemd/ipfs-accelerate-dashboard.service /etc/systemd/system/

# Reload systemd and enable services
sudo systemctl daemon-reload
sudo systemctl enable ipfs-accelerate.service
sudo systemctl enable ipfs-accelerate-dashboard.service

# Start them now
sudo systemctl start ipfs-accelerate.service
sudo systemctl start ipfs-accelerate-dashboard.service

# Check status
systemctl status ipfs-accelerate.service
systemctl status ipfs-accelerate-dashboard.service
```

## Configuration
- Units default to:
  - MCP API on port 9000, binds `0.0.0.0`.
  - Performance Dashboard on port 8080, binds `0.0.0.0`.
- Adjust environment or ports by editing the unit files.

## Logs
- All logs go to journald by default:
  - `sudo journalctl -u ipfs-accelerate -f`
  - `sudo journalctl -u ipfs-accelerate-dashboard -f`

## Firewall
```bash
sudo ufw allow 9000/tcp
sudo ufw allow 8080/tcp
```