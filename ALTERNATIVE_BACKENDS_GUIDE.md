# Alternative Cache Backends - ipfs-kit-py Integration

## Overview

When `libp2p-py` is not available (especially on Windows), you can use alternative backends provided by the `ipfs-kit-py` package to store and retrieve GitHub API cache data.

## Quick Start

```bash
# Install ipfs-kit-py
pip install ipfs-kit-py

# Run cross-platform test - it will automatically test available backends
python test_cross_platform_cache.py
```

## Supported Backends

### 1. Kubo (IPFS) Backend ⭐ (Recommended for Local)

**What it is:** Local IPFS daemon

**Pros:**
- Content-addressed storage
- Distributed by design
- No external dependencies
- Works offline

**Cons:**
- Requires IPFS daemon running
- Uses disk space
- Initial setup needed

**Setup:**

```bash
# Install IPFS (Kubo)
# Linux:
wget https://dist.ipfs.tech/kubo/v0.24.0/kubo_v0.24.0_linux-amd64.tar.gz
tar -xvzf kubo_v0.24.0_linux-amd64.tar.gz
cd kubo
sudo bash install.sh

# macOS:
brew install ipfs

# Windows:
# Download from https://dist.ipfs.tech/kubo/

# Initialize and start
ipfs init
ipfs daemon &
```

**Usage:**

```python
from ipfs_kit import IPFSKit

# Create Kubo backend
kit = IPFSKit(backend='kubo')

# Store cache data
cache_data = {
    "key": "repos/owner/name",
    "value": {"data": "..."},
    "timestamp": 1234567890
}
cid = kit.add_json(cache_data)
print(f"Stored with CID: {cid}")

# Retrieve cache data
retrieved = kit.get_json(cid)
print(f"Retrieved: {retrieved}")
```

**Environment Variables:**

```bash
# Optional: Custom IPFS API endpoint
export IPFS_API=/ip4/127.0.0.1/tcp/5001

# Use Kubo backend
export CACHE_BACKEND=kubo
```

### 2. Storacha (web3.storage) Backend ☁️ (Recommended for Cloud)

**What it is:** Managed IPFS service by Protocol Labs

**Pros:**
- No self-hosting required
- Built on IPFS
- Free tier available
- Automatic pinning

**Cons:**
- Requires internet connection
- Needs API token
- Rate limits on free tier

**Setup:**

```bash
# 1. Sign up at https://web3.storage
# 2. Create an API token
# 3. Set environment variable

export WEB3_STORAGE_TOKEN=your_token_here
# OR
export STORACHA_TOKEN=your_token_here
```

**Usage:**

```python
from ipfs_kit import IPFSKit
import os

# Create Storacha backend
token = os.environ['WEB3_STORAGE_TOKEN']
kit = IPFSKit(backend='storacha', token=token)

# Store cache data
cache_data = {"key": "test", "value": "data"}
cid = kit.add_json(cache_data)
print(f"Stored on web3.storage: {cid}")

# Retrieve cache data
retrieved = kit.get_json(cid)
```

**Environment Variables:**

```bash
# Required: API token
export WEB3_STORAGE_TOKEN=eyJhbG...your_token

# Use Storacha backend
export CACHE_BACKEND=storacha
```

**Get Token:**
1. Visit https://web3.storage
2. Sign up/login
3. Go to Account → Create API Token
4. Copy token and set environment variable

### 3. S3-Compatible Backend 📦 (AWS/MinIO/Backblaze)

**What it is:** S3-compatible object storage

**Pros:**
- Works with AWS S3, MinIO, Backblaze B2, etc.
- Well-understood caching patterns
- High availability
- Scalable

**Cons:**
- May have costs (AWS S3)
- Requires credentials
- Not content-addressed

**Setup:**

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export S3_BUCKET=github-api-cache

# Optional: Custom endpoint (for MinIO, B2, etc.)
export S3_ENDPOINT=https://s3.amazonaws.com
export S3_REGION=us-east-1
```

**Usage:**

```python
from ipfs_kit import IPFSKit
import os

# Create S3 backend
kit = IPFSKit(
    backend='s3',
    access_key=os.environ['AWS_ACCESS_KEY_ID'],
    secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    bucket=os.environ['S3_BUCKET'],
    endpoint=os.environ.get('S3_ENDPOINT'),  # Optional
    region=os.environ.get('S3_REGION', 'us-east-1')
)

# Store cache data
cache_data = {"key": "test", "value": "data"}
key = kit.add_json(cache_data)
print(f"Stored in S3: {key}")

# Retrieve cache data
retrieved = kit.get_json(key)
```

**MinIO Setup (Self-hosted S3):**

```bash
# Run MinIO locally
docker run -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"

# Configure
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export S3_BUCKET=cache
export S3_ENDPOINT=http://localhost:9000
```

## Integration with GitHub API Cache

### Option 1: Environment Variable Configuration

```bash
# Choose backend
export CACHE_BACKEND=kubo  # or storacha, s3

# Configure backend-specific settings
# For Kubo:
export IPFS_API=/ip4/127.0.0.1/tcp/5001

# For Storacha:
export WEB3_STORAGE_TOKEN=your_token

# For S3:
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export S3_BUCKET=cache

# Use cache normally
python your_script.py
```

### Option 2: Programmatic Configuration

```python
from ipfs_accelerate_py.github_cli.cache import configure_cache

# Configure with alternative backend
cache = configure_cache(
    enable_p2p=False,  # Disable P2P
    backend='kubo',    # Use Kubo instead
    backend_config={
        'api': '/ip4/127.0.0.1/tcp/5001'
    }
)

# Or with Storacha
cache = configure_cache(
    enable_p2p=False,
    backend='storacha',
    backend_config={
        'token': 'your_token'
    }
)

# Use cache normally
data = cache.get('key')
cache.put('key', data, ttl=300)
```

## Testing Alternative Backends

The cross-platform test automatically tests all available backends:

```bash
# Run test - it will try P2P first, then alternatives
python test_cross_platform_cache.py
```

**Output will show:**

```
Testing cache initialization...
P2P initialization error: libp2p not available
P2P not available, testing alternative backends...
Testing ipfs-kit-py alternative backends...
✅ ipfs-kit-py package available
Testing Kubo (IPFS) backend...
✅ Kubo backend works!
Testing Storacha (web3.storage) backend...
⚠️  Storacha token not set (WEB3_STORAGE_TOKEN or STORACHA_TOKEN)
Testing S3-compatible backend...
⚠️  S3 credentials not set

ALTERNATIVE BACKENDS AVAILABLE
✅ KUBO backend is working
```

## Platform Compatibility

| Backend | Linux | Windows | macOS | Docker |
|---------|-------|---------|-------|--------|
| Kubo | ✅ | ✅ | ✅ | ✅ |
| Storacha | ✅ | ✅ | ✅ | ✅ |
| S3 | ✅ | ✅ | ✅ | ✅ |

All backends work on all platforms!

## Performance Comparison

| Backend | Latency | Storage | Cost | Offline |
|---------|---------|---------|------|---------|
| Kubo | ~10ms | Local | Free | ✅ Yes |
| Storacha | ~100ms | Cloud | Free tier | ❌ No |
| S3 | ~50ms | Cloud | $0.023/GB | ❌ No |
| libp2p | ~5ms | Distributed | Free | ⚠️ Partial |

## Troubleshooting

### Kubo: "Connection refused"

```bash
# Check if IPFS daemon is running
ipfs id

# If not, start it
ipfs daemon &

# Check API endpoint
ipfs config Addresses.API
```

### Storacha: "Invalid token"

```bash
# Verify token is set
echo $WEB3_STORAGE_TOKEN

# Test token
curl -H "Authorization: Bearer $WEB3_STORAGE_TOKEN" \
  https://api.web3.storage/user/uploads
```

### S3: "Access denied"

```bash
# Check credentials
aws s3 ls s3://$S3_BUCKET

# Verify permissions
aws iam get-user
```

## Recommendations

### For Development (Local)
**Use Kubo**
- No external dependencies
- Works offline
- Fast
- Free

### For CI/CD (Cloud)
**Use Storacha**
- No infrastructure
- Free tier
- Simple setup
- Managed

### For Production (Enterprise)
**Use S3**
- High availability
- Well-understood
- Scalable
- Support available

## Next Steps

1. **Install ipfs-kit-py:**
   ```bash
   pip install ipfs-kit-py
   ```

2. **Choose and configure backend:**
   - Kubo: `ipfs daemon &`
   - Storacha: Get token from web3.storage
   - S3: Set AWS credentials

3. **Run test:**
   ```bash
   python test_cross_platform_cache.py
   ```

4. **Update your config:**
   ```bash
   export CACHE_BACKEND=kubo  # or storacha, s3
   ```

5. **Use cache normally!**

## Support

- **ipfs-kit-py:** https://github.com/endomorphosis/ipfs-kit-py
- **Kubo (IPFS):** https://docs.ipfs.tech/install/
- **Storacha:** https://web3.storage/docs/
- **Cross-platform test:** `python test_cross_platform_cache.py`
