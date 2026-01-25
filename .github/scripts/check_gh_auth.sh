#!/bin/bash

# GitHub CLI Authentication Preflight Check
# Validates that gh CLI is authenticated and tokens are valid before workflow operations

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Resolve repo root so this script works no matter the current working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GH_API_CACHED=(python3 "$REPO_ROOT/tools/gh_api_cached.py")

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Check if gh CLI is installed
check_gh_installed() {
    if ! command -v gh &> /dev/null; then
        log_error "GitHub CLI (gh) is not installed"
        return 1
    fi
    
    local version=$(gh --version | head -n1)
    log_info "GitHub CLI found: $version"
    return 0
}

# Check if GITHUB_TOKEN or GH_TOKEN is available
check_token_available() {
    if [ -n "$GH_TOKEN" ]; then
        log_info "GH_TOKEN environment variable is set"
        return 0
    elif [ -n "$GITHUB_TOKEN" ]; then
        log_info "GITHUB_TOKEN environment variable is set (will be used as GH_TOKEN)"
        export GH_TOKEN="$GITHUB_TOKEN"
        return 0
    else
        log_error "No authentication token found (GH_TOKEN or GITHUB_TOKEN)"
        return 1
    fi
}

# Validate token by making a test API call
validate_token() {
    log_info "Validating GitHub token..."
    
    # Test authentication with a simple API call
    if gh auth status &> /dev/null; then
        log_info "✓ GitHub authentication is valid"
        
        # Get user info to confirm
        local user=$(${GH_API_CACHED[@]} user --jq '.login' 2>/dev/null || echo "unknown")
        log_info "Authenticated as: $user"
        
        return 0
    else
        log_error "✗ GitHub authentication failed"
        log_error "Token may be expired or invalid"
        return 1
    fi
}

# Check rate limit status
check_rate_limit() {
    log_info "Checking API rate limits..."
    
    local rate_limit=$(${GH_API_CACHED[@]} rate_limit 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        local remaining=$(echo "$rate_limit" | jq -r '.rate.remaining' 2>/dev/null || echo "unknown")
        local limit=$(echo "$rate_limit" | jq -r '.rate.limit' 2>/dev/null || echo "unknown")
        local reset=$(echo "$rate_limit" | jq -r '.rate.reset' 2>/dev/null || echo "unknown")
        
        if [ "$remaining" != "unknown" ]; then
            log_info "Rate limit: $remaining / $limit remaining"
            
            if [ "$remaining" -lt 100 ]; then
                log_warn "⚠ Low rate limit remaining: $remaining requests"
                
                if [ "$reset" != "unknown" ]; then
                    local reset_time=$(date -d "@$reset" 2>/dev/null || echo "unknown")
                    log_warn "Rate limit resets at: $reset_time"
                fi
            fi
        fi
        
        return 0
    else
        log_warn "Could not check rate limit (may not affect operation)"
        return 0
    fi
}

# Check token scopes/permissions
check_token_scopes() {
    log_info "Checking token scopes..."
    
    # Get the scopes from the token
    local scopes=$(${GH_API_CACHED[@]} /user -i 2>&1 | grep -i "x-oauth-scopes:" | cut -d: -f2- | tr -d ' ')
    
    if [ -n "$scopes" ]; then
        log_info "Token scopes: $scopes"
        
        # Check for required scopes
        local has_repo=false
        local has_workflow=false
        
        if echo "$scopes" | grep -q "repo"; then
            has_repo=true
        fi
        
        if echo "$scopes" | grep -q "workflow"; then
            has_workflow=true
        fi
        
        if [ "$has_repo" = true ]; then
            log_info "✓ 'repo' scope present"
        else
            log_warn "⚠ 'repo' scope missing (may be needed for some operations)"
        fi
        
        if [ "$has_workflow" = true ]; then
            log_info "✓ 'workflow' scope present"
        else
            log_warn "⚠ 'workflow' scope missing (may be needed for workflow operations)"
        fi
    else
        log_debug "Could not determine token scopes (proceeding anyway)"
    fi
    
    return 0
}

# Main preflight check
main() {
    log_info "GitHub CLI Authentication Preflight Check"
    log_info "=========================================="
    echo ""
    
    local failed=false
    
    # Step 1: Check if gh is installed
    if ! check_gh_installed; then
        log_error "Preflight check failed: gh CLI not installed"
        exit 1
    fi
    echo ""
    
    # Step 2: Check if token is available
    if ! check_token_available; then
        log_error "Preflight check failed: no authentication token"
        log_error ""
        log_error "To fix this, ensure one of the following:"
        log_error "  1. Set GITHUB_TOKEN environment variable"
        log_error "  2. Set GH_TOKEN environment variable"
        log_error "  3. Run 'gh auth login' to authenticate"
        exit 1
    fi
    echo ""
    
    # Step 3: Validate token
    if ! validate_token; then
        log_error "Preflight check failed: token validation failed"
        log_error ""
        log_error "To fix this:"
        log_error "  1. Check if your token has expired"
        log_error "  2. Generate a new token at: https://github.com/settings/tokens"
        log_error "  3. Or run 'gh auth login' to re-authenticate"
        exit 1
    fi
    echo ""
    
    # Step 4: Check rate limits (non-fatal)
    check_rate_limit
    echo ""
    
    # Step 5: Check token scopes (non-fatal)
    check_token_scopes
    echo ""
    
    log_info "=========================================="
    log_info "✓ All preflight checks passed"
    log_info ""
    
    return 0
}

# Show usage if requested
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0"
    echo ""
    echo "GitHub CLI Authentication Preflight Check"
    echo ""
    echo "This script validates GitHub CLI authentication before running workflows."
    echo "It checks for:"
    echo "  - gh CLI installation"
    echo "  - Valid authentication token (GH_TOKEN or GITHUB_TOKEN)"
    echo "  - Token validity and expiration"
    echo "  - API rate limits"
    echo "  - Token scopes/permissions"
    echo ""
    echo "Environment variables:"
    echo "  GH_TOKEN       - GitHub personal access token"
    echo "  GITHUB_TOKEN   - GitHub token (will be copied to GH_TOKEN if set)"
    echo ""
    echo "Exit codes:"
    echo "  0 - All checks passed"
    echo "  1 - Preflight check failed"
    exit 0
fi

main "$@"
