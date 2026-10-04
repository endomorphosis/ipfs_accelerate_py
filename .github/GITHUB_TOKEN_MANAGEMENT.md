# GitHub Token Authentication & Rotation System

## Overview

This system ensures GitHub CLI operations in workflows have valid authentication tokens and provides tools for token rotation when they expire.

## Components

### 1. Preflight Check Script (`check_gh_auth.sh`)

**Purpose**: Validates GitHub authentication before workflow operations

**Location**: `.github/scripts/check_gh_auth.sh`

**Features**:
- ✅ Verifies gh CLI is installed
- ✅ Checks for GH_TOKEN or GITHUB_TOKEN environment variables
- ✅ Validates token is active and not expired
- ✅ Checks API rate limits
- ✅ Verifies token has required scopes (repo, workflow)

**Usage**:
```bash
# Run preflight check
.github/scripts/check_gh_auth.sh

# In workflows (automatic)
- name: GitHub CLI Authentication Preflight Check
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: bash .github/scripts/check_gh_auth.sh
```

**Exit Codes**:
- `0` - All checks passed
- `1` - Authentication failed (missing/invalid token)

### 2. Token Rotation Script (`rotate_gh_token.sh`)

**Purpose**: Helps rotate expired or invalid GitHub tokens

**Location**: `.github/scripts/rotate_gh_token.sh`

**Modes**:

#### Interactive Mode (Default)
```bash
.github/scripts/rotate_gh_token.sh
# or
.github/scripts/rotate_gh_token.sh --interactive
```

Provides menu with options:
1. Browser authentication (recommended)
2. Manual token entry
3. Use existing GITHUB_TOKEN environment variable

#### Automatic Mode
```bash
.github/scripts/rotate_gh_token.sh --auto ghp_your_new_token_here
```

#### Export Mode
```bash
.github/scripts/rotate_gh_token.sh --export
```
Exports current token to GH_TOKEN and GITHUB_TOKEN environment variables

#### Save Mode
```bash
.github/scripts/rotate_gh_token.sh --save
```
Saves token to `~/.github_token` with restricted permissions (600)

## Workflow Integration

### Auto-Heal Workflow
The auto-heal workflow includes automatic preflight checks:

```yaml
- name: GitHub CLI Authentication Preflight Check
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    # Install gh CLI if needed
    if ! command -v gh &> /dev/null; then
      # ... installation code ...
    fi
    
    # Run preflight check
    bash .github/scripts/check_gh_auth.sh
```

### Cleanup Workflow
The cleanup workflow includes optional preflight checks when gh CLI is available.

## Token Management Best Practices

### Creating a Personal Access Token

1. Go to https://github.com/settings/tokens/new
2. Select scopes:
   - ✅ `repo` - Full control of private repositories
   - ✅ `workflow` - Update GitHub Action workflows
   - ✅ `admin:org` - (if using organization runners)
3. Set expiration (recommended: 90 days)
4. Generate token and save securely

### Setting Up Authentication

#### For Local Development
```bash
# Method 1: Browser authentication
gh auth login

# Method 2: Token authentication
export GITHUB_TOKEN="ghp_your_token_here"
echo "$GITHUB_TOKEN" | gh auth login --with-token

# Method 3: Use rotation script
.github/scripts/rotate_gh_token.sh
```

#### For CI/CD (GitHub Actions)
Tokens are automatically provided via `secrets.GITHUB_TOKEN` - no setup needed!

#### For Self-Hosted Runners
```bash
# Set in runner environment
echo "GITHUB_TOKEN=ghp_your_token_here" >> ~/.bashrc
source ~/.bashrc

# Or use the rotation script
.github/scripts/rotate_gh_token.sh --auto ghp_your_token_here
```

### Token Rotation Process

When a token expires:

1. **Detect the issue**:
   ```
   Error: Process completed with exit code 1.
   The value of the GH_TOKEN environment variable is being used for authentication.
   ```

2. **Generate new token**:
   - Visit https://github.com/settings/tokens
   - Generate new token with same scopes
   - Copy the token

3. **Rotate the token**:
   ```bash
   # Interactive (recommended)
   .github/scripts/rotate_gh_token.sh
   
   # Or automatic
   .github/scripts/rotate_gh_token.sh --auto ghp_new_token_here
   ```

4. **Verify**:
   ```bash
   .github/scripts/check_gh_auth.sh
   ```

5. **Update secrets** (if using GitHub Actions):
   - Go to repository Settings → Secrets and variables → Actions
   - Update `GITHUB_TOKEN` or add custom token secret
   - Re-run failed workflow

## Troubleshooting

### "No authentication token found"
```bash
# Fix: Export token to environment
export GITHUB_TOKEN="ghp_your_token_here"
# or
export GH_TOKEN="ghp_your_token_here"
```

### "Token validation failed"
```bash
# Check if token is expired
gh auth status

# Rotate to new token
.github/scripts/rotate_gh_token.sh
```

### "Low rate limit remaining"
```bash
# Check rate limit status
gh api rate_limit

# Wait for reset or use different token
# Reset time shown in preflight check output
```

### "Required scope missing"
Token needs additional permissions:
1. Create new token with required scopes
2. Rotate using rotation script

## Monitoring

### Check Authentication Status
```bash
# Quick check
gh auth status

# Detailed preflight check
.github/scripts/check_gh_auth.sh
```

### Check Rate Limits
```bash
# Via preflight check (recommended)
.github/scripts/check_gh_auth.sh

# Via gh CLI
gh api rate_limit
```

### View Token Scopes
```bash
gh api -i /user | grep "x-oauth-scopes:"
```

## Security Considerations

### Token Storage
- ✅ Never commit tokens to git
- ✅ Use environment variables
- ✅ Set restrictive file permissions (600) if storing in files
- ✅ Use GitHub Secrets for CI/CD
- ❌ Don't log tokens in workflow outputs
- ❌ Don't share tokens in plain text

### Token Expiration
- Set appropriate expiration periods (30-90 days recommended)
- Monitor for expiration warnings
- Rotate proactively before expiration
- Keep backup tokens ready for rotation

### Scope Minimization
Only grant scopes actually needed:
- `repo` - For repository operations
- `workflow` - For workflow modifications
- `admin:org` - Only if managing organization resources

## Automated Monitoring

The preflight check automatically monitors:
- Token validity
- Rate limit status
- Scope availability
- Authentication status

Failed checks will cause workflows to fail early with clear error messages.

## Quick Reference

```bash
# Check authentication
.github/scripts/check_gh_auth.sh

# Rotate token (interactive)
.github/scripts/rotate_gh_token.sh

# Rotate token (automatic)
.github/scripts/rotate_gh_token.sh --auto ghp_new_token

# Export token to environment
.github/scripts/rotate_gh_token.sh --export

# Save token to file
.github/scripts/rotate_gh_token.sh --save

# Manual gh CLI login
gh auth login

# Check gh status
gh auth status

# Check rate limits
gh api rate_limit
```

## Support

If issues persist:
1. Check GitHub Status: https://www.githubstatus.com/
2. Review token permissions at https://github.com/settings/tokens
3. Verify runner/workflow permissions
4. Check workflow run logs for detailed errors

---

**Last Updated**: November 7, 2025  
**Version**: 1.0.0
