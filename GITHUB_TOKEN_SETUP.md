# GitHub Token Configuration

This project uses GitHub Personal Access Tokens (PAT) for:
- GitHub CLI authentication
- GitHub Copilot integration  
- GitHub Actions autoscaler
- API access for workflow management

## Quick Setup

```bash
# Run the interactive setup script
bash scripts/setup-github-token.sh
```

## Manual Setup

### 1. Generate a Personal Access Token

Visit: https://github.com/settings/tokens/new

**Required scopes**:
- ✅ `repo` - Full control of private repositories
- ✅ `workflow` - Update GitHub Action workflows
- ✅ `admin:org` - Full control of orgs and teams (if using org runners)
- ✅ `copilot` - GitHub Copilot access

**Recommended name**: `ipfs_accelerate_py_token`

Click "Generate token" and copy the token (starts with `ghp_` or `github_pat_`)

### 2. Configure the Token

**Option A: Using gh CLI (Recommended)**
```bash
# Authenticate interactively
gh auth login -h github.com

# Or use token directly
echo "YOUR_TOKEN_HERE" | gh auth login --with-token
```

**Option B: Environment Variables**
```bash
# Add to ~/.bashrc or ~/.zshrc
export GITHUB_TOKEN="ghp_YOUR_TOKEN_HERE"
export GH_TOKEN="ghp_YOUR_TOKEN_HERE"

# Reload shell
source ~/.bashrc
```

**Option C: .env File (For Autoscaler)**
```bash
# Create .env file (already in .gitignore)
echo "GITHUB_TOKEN=ghp_YOUR_TOKEN_HERE" > .env
chmod 600 .env
```

## GitHub Copilot Setup

### In VS Code

1. **Install Extension**
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search "GitHub Copilot"
   - Click Install

2. **Sign In**
   - Click on Copilot icon in status bar
   - Sign in with GitHub
   - Authorize GitHub Copilot

3. **Verify**
   - Open a Python file
   - Start typing - you should see Copilot suggestions
   - Press Tab to accept suggestions

### For GitHub Copilot CLI

```bash
# Install Copilot CLI
gh extension install github/gh-copilot

# Use it
gh copilot suggest "how to fix permission errors in github actions"
gh copilot explain "git checkout -b feature/new-branch"
```

## Verify Setup

```bash
# Check gh CLI authentication
gh auth status

# Test API access
gh api user

# Check token scopes
gh api user -i | grep x-oauth-scopes

# Test with repo
gh repo view endomorphosis/ipfs_accelerate_py
```

## Security Best Practices

### ✅ DO:
- Generate tokens with minimum required scopes
- Use different tokens for different purposes
- Store tokens securely (gh CLI credential store or env vars)
- Add `.env` to `.gitignore`
- Set `.env` file permissions to 600 (owner read/write only)
- Rotate tokens periodically (every 90 days recommended)
- Revoke tokens immediately if compromised

### ❌ DON'T:
- Commit tokens to git repositories
- Share tokens via email, chat, or other insecure channels
- Use the same token for multiple projects/machines
- Grant more scopes than necessary
- Leave tokens with unlimited expiration

## Token Expiration

If your token expires or becomes invalid:

```bash
# Re-authenticate
gh auth refresh -h github.com

# Or login again
bash scripts/setup-github-token.sh
```

## Troubleshooting

### "Token is invalid" Error

```bash
# Check if token works directly
curl -H "Authorization: token YOUR_TOKEN" https://api.github.com/user

# Re-authenticate
gh auth logout
gh auth login
```

### Copilot Not Working

1. Check subscription: https://github.com/settings/copilot
2. Verify VS Code extension is enabled
3. Check VS Code settings: `github.copilot.enable`
4. Restart VS Code

### Autoscaler Can't Access API

```bash
# Verify token is in environment
echo $GITHUB_TOKEN

# Restart service
sudo systemctl restart github-autoscaler@barberb.service

# Check logs
sudo journalctl -u github-autoscaler@barberb.service -f
```

## Services Using the Token

### GitHub Autoscaler
- **Location**: `/etc/systemd/system/github-autoscaler@.service`
- **Token Source**: Environment variable or gh CLI
- **Restart**: `sudo systemctl restart github-autoscaler@barberb.service`

### Runner Permission Fix Timer
- **Location**: `/etc/systemd/system/runner-permission-fix.timer`
- **Token**: Uses gh CLI credentials
- **Check**: `systemctl status runner-permission-fix.timer`

### Workflows
- **Token**: Automatically provided by GitHub Actions as `${{ secrets.GITHUB_TOKEN }}`
- **No setup needed**: GitHub injects this automatically

## Token Scopes Reference

| Scope | Purpose | Required? |
|-------|---------|-----------|
| `repo` | Access repositories | ✅ Yes |
| `workflow` | Manage workflows | ✅ Yes |
| `admin:org` | Manage org runners | ✅ Yes (for org) |
| `copilot` | GitHub Copilot | ✅ Yes (for Copilot) |
| `read:org` | Read org data | ⚠️ Optional |
| `user` | Read user data | ⚠️ Optional |

## Additional Resources

- [GitHub PAT Documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
- [GitHub CLI Auth](https://cli.github.com/manual/gh_auth_login)
- [GitHub Copilot Docs](https://docs.github.com/en/copilot)
- [Fine-grained PAT](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token#creating-a-fine-grained-personal-access-token)
