#!/bin/bash

# GitHub Personal Access Token (PAT) Setup Script
# This script helps you securely configure your GitHub PAT for:
# - GitHub CLI (gh)
# - GitHub Copilot integration
# - GitHub Actions autoscaler
# - API access

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔐 GitHub Personal Access Token Setup${NC}"
echo "=========================================="
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}❌ GitHub CLI (gh) is not installed${NC}"
    echo "   Run: bash scripts/update-gh-cli.sh"
    exit 1
fi

echo -e "${GREEN}✅ GitHub CLI found: $(gh --version | head -1)${NC}"
echo ""

# Function to check if token is valid
check_token() {
    local token="$1"
    if [ -z "$token" ]; then
        return 1
    fi
    
    # Test token by making API call
    local response=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: token $token" https://api.github.com/user)
    if [ "$response" = "200" ]; then
        return 0
    else
        return 1
    fi
}

# Function to get token scopes
get_token_scopes() {
    local token="$1"
    curl -s -H "Authorization: token $token" https://api.github.com/user -I | grep -i "x-oauth-scopes" | cut -d':' -f2 | xargs
}

# Check current authentication status
echo "📊 Current Authentication Status:"
echo "-----------------------------------"
if gh auth status 2>&1 | grep -q "Logged in"; then
    echo -e "${GREEN}✅ Already authenticated with gh CLI${NC}"
    gh auth status 2>&1 | head -10
    echo ""
    read -p "Do you want to reconfigure? (y/N): " reconfigure
    if [[ ! "$reconfigure" =~ ^[Yy]$ ]]; then
        echo "Configuration unchanged."
        exit 0
    fi
else
    echo -e "${YELLOW}⚠️  Not currently authenticated${NC}"
fi

echo ""
echo "🎯 Setup Options:"
echo "-----------------------------------"
echo "1) Interactive login (recommended for first-time setup)"
echo "2) Paste PAT token (for automation/scripts)"
echo "3) Load PAT from file"
echo "4) Set PAT as environment variable only"
echo ""
read -p "Choose option (1-4): " option

case $option in
    1)
        echo ""
        echo "🌐 Starting interactive login..."
        echo "   This will open a browser for OAuth authentication"
        echo ""
        gh auth login -h github.com -p https -w
        ;;
    
    2)
        echo ""
        echo "📝 Paste your Personal Access Token:"
        echo "   (Get one at: https://github.com/settings/tokens/new?scopes=repo,workflow,admin:org,copilot)"
        echo ""
        read -sp "Token: " token
        echo ""
        
        if [ -z "$token" ]; then
            echo -e "${RED}❌ No token provided${NC}"
            exit 1
        fi
        
        echo ""
        echo "🔍 Validating token..."
        if check_token "$token"; then
            scopes=$(get_token_scopes "$token")
            echo -e "${GREEN}✅ Token is valid${NC}"
            echo "   Scopes: $scopes"
            echo ""
            
            # Authenticate with gh CLI
            echo "$token" | gh auth login --with-token -h github.com
            echo -e "${GREEN}✅ Authenticated with gh CLI${NC}"
            
            # Optionally save to .env file
            read -p "Save token to .env file for autoscaler? (y/N): " save_env
            if [[ "$save_env" =~ ^[Yy]$ ]]; then
                if [ -f .env ]; then
                    # Backup existing .env
                    cp .env .env.backup
                fi
                
                # Update or append GITHUB_TOKEN
                if grep -q "^GITHUB_TOKEN=" .env 2>/dev/null; then
                    sed -i "s|^GITHUB_TOKEN=.*|GITHUB_TOKEN=$token|" .env
                else
                    echo "GITHUB_TOKEN=$token" >> .env
                fi
                
                chmod 600 .env
                echo -e "${GREEN}✅ Token saved to .env (chmod 600)${NC}"
                
                # Add .env to .gitignore if not already there
                if [ -f .gitignore ] && ! grep -q "^\.env$" .gitignore; then
                    echo ".env" >> .gitignore
                    echo -e "${GREEN}✅ Added .env to .gitignore${NC}"
                fi
            fi
        else
            echo -e "${RED}❌ Token is invalid or has insufficient permissions${NC}"
            echo "   Required scopes: repo, workflow, admin:org, copilot"
            exit 1
        fi
        ;;
    
    3)
        echo ""
        read -p "Enter path to token file: " token_file
        
        if [ ! -f "$token_file" ]; then
            echo -e "${RED}❌ File not found: $token_file${NC}"
            exit 1
        fi
        
        token=$(cat "$token_file" | tr -d '[:space:]')
        
        if check_token "$token"; then
            echo -e "${GREEN}✅ Token is valid${NC}"
            echo "$token" | gh auth login --with-token -h github.com
            echo -e "${GREEN}✅ Authenticated with gh CLI${NC}"
        else
            echo -e "${RED}❌ Token is invalid${NC}"
            exit 1
        fi
        ;;
    
    4)
        echo ""
        echo "📝 Paste your Personal Access Token:"
        read -sp "Token: " token
        echo ""
        
        if check_token "$token"; then
            echo -e "${GREEN}✅ Token is valid${NC}"
            echo ""
            echo "Add this to your shell profile (~/.bashrc or ~/.zshrc):"
            echo ""
            echo -e "${YELLOW}export GITHUB_TOKEN=\"$token\"${NC}"
            echo -e "${YELLOW}export GH_TOKEN=\"$token\"${NC}"
            echo ""
            echo "Then run: source ~/.bashrc"
        else
            echo -e "${RED}❌ Token is invalid${NC}"
            exit 1
        fi
        ;;
    
    *)
        echo -e "${RED}❌ Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo "🎉 Setup Complete!"
echo "-----------------------------------"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Verify authentication:"
echo "   gh auth status"
echo ""
echo "2. Test API access:"
echo "   gh repo view endomorphosis/ipfs_accelerate_py"
echo ""
echo "3. For GitHub Copilot in VS Code:"
echo "   - Install GitHub Copilot extension"
echo "   - Sign in with your GitHub account"
echo "   - Copilot will use your authenticated account"
echo ""
echo "4. For automation/scripts:"
echo "   - The token is now configured in gh CLI"
echo "   - Use: gh api /user"
echo "   - Or set: export GH_TOKEN=\$(gh auth token)"
echo ""
echo "5. Restart services to pick up new token:"
echo "   sudo systemctl restart github-autoscaler@barberb.service"
echo ""

# Show final status
echo "📊 Final Status:"
echo "-----------------------------------"
gh auth status 2>&1 || true
