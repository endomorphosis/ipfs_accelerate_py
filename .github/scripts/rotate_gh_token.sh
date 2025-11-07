#!/bin/bash

# GitHub Token Rotation Helper
# Helps rotate expired or invalid GitHub tokens for workflow automation

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check current auth status
check_current_auth() {
    log_info "Checking current authentication status..."
    
    if gh auth status &> /dev/null; then
        log_info "✓ Currently authenticated"
        gh auth status 2>&1 | grep -E "Logged in|Token:" || true
        return 0
    else
        log_warn "✗ Not currently authenticated or token is invalid"
        return 1
    fi
}

# Interactive token rotation
rotate_token_interactive() {
    log_info "GitHub Token Rotation - Interactive Mode"
    log_info "========================================"
    echo ""
    
    log_step "Step 1: Check current authentication"
    check_current_auth || true
    echo ""
    
    log_step "Step 2: Choose authentication method"
    echo ""
    echo "Select how you want to authenticate:"
    echo "  1) Browser authentication (recommended)"
    echo "  2) Enter personal access token manually"
    echo "  3) Use existing GITHUB_TOKEN environment variable"
    echo ""
    read -p "Enter choice [1-3]: " choice
    echo ""
    
    case $choice in
        1)
            log_info "Starting browser-based authentication..."
            gh auth login
            ;;
        2)
            log_info "Manual token entry"
            echo ""
            log_warn "You need to create a personal access token at:"
            log_warn "https://github.com/settings/tokens/new"
            echo ""
            log_info "Required scopes:"
            log_info "  ✓ repo (Full control of private repositories)"
            log_info "  ✓ workflow (Update GitHub Action workflows)"
            log_info "  ✓ admin:org (if using organization runners)"
            echo ""
            read -sp "Enter your personal access token: " token
            echo ""
            
            if [ -z "$token" ]; then
                log_error "No token provided"
                return 1
            fi
            
            # Authenticate with the token
            echo "$token" | gh auth login --with-token
            ;;
        3)
            if [ -z "$GITHUB_TOKEN" ]; then
                log_error "GITHUB_TOKEN environment variable is not set"
                return 1
            fi
            
            log_info "Using GITHUB_TOKEN environment variable"
            echo "$GITHUB_TOKEN" | gh auth login --with-token
            ;;
        *)
            log_error "Invalid choice"
            return 1
            ;;
    esac
    
    echo ""
    log_step "Step 3: Verify new authentication"
    
    if gh auth status &> /dev/null; then
        log_info "✓ Authentication successful!"
        gh auth status
        echo ""
        log_info "Token has been rotated successfully"
        return 0
    else
        log_error "✗ Authentication failed"
        return 1
    fi
}

# Non-interactive token rotation (for automation)
rotate_token_auto() {
    local token="$1"
    
    if [ -z "$token" ]; then
        log_error "No token provided for automatic rotation"
        return 1
    fi
    
    log_info "Rotating token (non-interactive mode)..."
    
    # Logout first
    gh auth logout --hostname github.com 2>/dev/null || true
    
    # Login with new token
    if echo "$token" | gh auth login --with-token; then
        log_info "✓ Token rotation successful"
        return 0
    else
        log_error "✗ Token rotation failed"
        return 1
    fi
}

# Export token for use in scripts
export_token() {
    log_info "Exporting token to environment..."
    
    local token=$(gh auth token 2>/dev/null)
    
    if [ -n "$token" ]; then
        export GH_TOKEN="$token"
        export GITHUB_TOKEN="$token"
        
        log_info "✓ Token exported to GH_TOKEN and GITHUB_TOKEN"
        log_info "You can now use GitHub CLI in this shell session"
        echo ""
        echo "To persist for this session, run:"
        echo "  export GH_TOKEN=\"\$(gh auth token)\""
        echo "  export GITHUB_TOKEN=\"\$(gh auth token)\""
        
        return 0
    else
        log_error "Could not retrieve token"
        return 1
    fi
}

# Save token to a secure location
save_token_secure() {
    local token_file="$HOME/.github_token"
    
    log_warn "Saving token to file: $token_file"
    log_warn "⚠ This file will contain sensitive credentials!"
    echo ""
    read -p "Continue? [y/N]: " confirm
    
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        log_info "Cancelled"
        return 1
    fi
    
    local token=$(gh auth token 2>/dev/null)
    
    if [ -z "$token" ]; then
        log_error "No token available to save"
        return 1
    fi
    
    # Save with restricted permissions
    echo "$token" > "$token_file"
    chmod 600 "$token_file"
    
    log_info "✓ Token saved to $token_file (permissions: 600)"
    echo ""
    log_info "To use this token later:"
    echo "  export GITHUB_TOKEN=\$(cat $token_file)"
    echo "  export GH_TOKEN=\$(cat $token_file)"
}

# Main function
main() {
    local mode="${1:-interactive}"
    
    case "$mode" in
        -i|--interactive|interactive)
            rotate_token_interactive
            ;;
        -a|--auto|auto)
            shift
            rotate_token_auto "$1"
            ;;
        -e|--export|export)
            export_token
            ;;
        -s|--save|save)
            save_token_secure
            ;;
        -h|--help|help)
            echo "Usage: $0 [mode] [options]"
            echo ""
            echo "GitHub Token Rotation Helper"
            echo ""
            echo "Modes:"
            echo "  -i, --interactive    Interactive token rotation (default)"
            echo "  -a, --auto TOKEN     Automatic rotation with provided token"
            echo "  -e, --export         Export current token to environment"
            echo "  -s, --save           Save token to ~/.github_token"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Interactive mode"
            echo "  $0 --auto ghp_xxxxx          # Rotate to specific token"
            echo "  $0 --export                  # Export to environment vars"
            echo "  $0 --save                    # Save to file"
            echo ""
            exit 0
            ;;
        *)
            log_error "Unknown mode: $mode"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
}

main "$@"
