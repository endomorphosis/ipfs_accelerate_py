#!/bin/bash
# Enterprise SSL/TLS Setup Script for IPFS Accelerate Python
# Automated SSL certificate management with enterprise features

set -euo pipefail

# Configuration
DOMAIN=${DOMAIN:-"ipfs-accelerate.local"}
ENVIRONMENT=${ENVIRONMENT:-"development"}
CERT_DIR="/etc/ssl/ipfs-accelerate"
BACKUP_DIR="/etc/ssl/backups"
LOG_FILE="/var/log/ipfs-accelerate-ssl.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    log "ERROR: $1"
}

success() {
    echo -e "${GREEN}SUCCESS: $1${NC}"
    log "SUCCESS: $1"
}

warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
    log "WARNING: $1"
}

info() {
    echo -e "${BLUE}INFO: $1${NC}"
    log "INFO: $1"
}

# Create necessary directories
create_directories() {
    info "Creating SSL directories..."
    sudo mkdir -p "$CERT_DIR" "$BACKUP_DIR" "$(dirname "$LOG_FILE")"
    sudo chmod 700 "$CERT_DIR" "$BACKUP_DIR"
}

# Generate self-signed certificate for development
generate_self_signed_cert() {
    info "Generating self-signed certificate for development..."
    
    local cert_file="$CERT_DIR/dev-cert.pem"
    local key_file="$CERT_DIR/dev-key.pem"
    
    # Generate private key
    openssl genrsa -out "$key_file" 2048
    chmod 600 "$key_file"
    
    # Generate certificate
    openssl req -new -x509 -key "$key_file" -out "$cert_file" -days 365 \
        -subj "/C=US/ST=CA/L=San Francisco/O=IPFS Accelerate/OU=Development/CN=$DOMAIN"
    
    chmod 644 "$cert_file"
    
    success "Self-signed certificate generated: $cert_file"
}

# Setup Let's Encrypt certificate for production
setup_letsencrypt() {
    info "Setting up Let's Encrypt certificate..."
    
    # Install certbot if not present
    if ! command -v certbot &> /dev/null; then
        info "Installing certbot..."
        sudo apt-get update && sudo apt-get install -y certbot python3-certbot-nginx
    fi
    
    # Generate certificate
    sudo certbot certonly --standalone \
        --non-interactive \
        --agree-tos \
        --email "admin@$DOMAIN" \
        --domains "$DOMAIN" \
        --cert-name "ipfs-accelerate"
    
    # Setup auto-renewal
    echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
    
    success "Let's Encrypt certificate configured for $DOMAIN"
}

# Validate SSL configuration
validate_ssl_config() {
    info "Validating SSL configuration..."
    
    local cert_file="$1"
    local key_file="$2"
    
    # Check certificate validity
    if openssl x509 -in "$cert_file" -text -noout > /dev/null 2>&1; then
        success "Certificate is valid"
        
        # Get certificate details
        local expiry_date=$(openssl x509 -in "$cert_file" -enddate -noout | cut -d= -f2)
        local subject=$(openssl x509 -in "$cert_file" -subject -noout | cut -d= -f2-)
        
        info "Certificate Subject: $subject"
        info "Certificate Expires: $expiry_date"
        
        # Check if certificate is expiring soon (30 days)
        local expiry_epoch=$(date -d "$expiry_date" +%s)
        local current_epoch=$(date +%s)
        local days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
        
        if [ "$days_until_expiry" -lt 30 ]; then
            warning "Certificate expires in $days_until_expiry days - renewal recommended"
        else
            info "Certificate valid for $days_until_expiry days"
        fi
    else
        error "Certificate validation failed"
        return 1
    fi
    
    # Check private key
    if openssl rsa -in "$key_file" -check -noout > /dev/null 2>&1; then
        success "Private key is valid"
    else
        error "Private key validation failed"
        return 1
    fi
    
    # Check key-certificate pair match
    local cert_modulus=$(openssl x509 -noout -modulus -in "$cert_file" | openssl md5)
    local key_modulus=$(openssl rsa -noout -modulus -in "$key_file" | openssl md5)
    
    if [ "$cert_modulus" = "$key_modulus" ]; then
        success "Certificate and private key match"
    else
        error "Certificate and private key do not match"
        return 1
    fi
}

# Setup Nginx SSL configuration
setup_nginx_ssl() {
    info "Setting up Nginx SSL configuration..."
    
    local nginx_config="/etc/nginx/sites-available/ipfs-accelerate-ssl"
    
    cat > "$nginx_config" << 'EOF'
server {
    listen 80;
    server_name ipfs-accelerate.local;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ipfs-accelerate.local;

    # SSL Configuration
    ssl_certificate /etc/ssl/ipfs-accelerate/dev-cert.pem;
    ssl_certificate_key /etc/ssl/ipfs-accelerate/dev-key.pem;
    
    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    
    # Application proxy
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://localhost:8000/health;
        access_log off;
    }
}
EOF

    sudo ln -sf "$nginx_config" /etc/nginx/sites-enabled/
    
    success "Nginx SSL configuration created"
}

# Create SSL certificates directory structure
create_cert_structure() {
    info "Creating certificate directory structure..."
    
    mkdir -p deployments/certs
    mkdir -p deployments/ssl-scripts
    
    # Create certificate generation script
    cat > deployments/ssl-scripts/generate_dev_certs.sh << 'EOF'
#!/bin/bash
# Generate development certificates

CERT_DIR="./deployments/certs"
mkdir -p "$CERT_DIR"

# Generate CA private key
openssl genrsa -out "$CERT_DIR/ca-key.pem" 4096

# Generate CA certificate
openssl req -new -x509 -days 365 -key "$CERT_DIR/ca-key.pem" \
    -sha256 -out "$CERT_DIR/ca-cert.pem" \
    -subj "/C=US/ST=CA/L=San Francisco/O=IPFS Accelerate Dev/CN=IPFS Accelerate Dev CA"

# Generate server private key
openssl genrsa -out "$CERT_DIR/server-key.pem" 4096

# Generate server certificate signing request
openssl req -subj "/C=US/ST=CA/L=San Francisco/O=IPFS Accelerate/CN=localhost" \
    -sha256 -new -key "$CERT_DIR/server-key.pem" -out "$CERT_DIR/server.csr"

# Generate server certificate
openssl x509 -req -days 365 -sha256 \
    -in "$CERT_DIR/server.csr" \
    -CA "$CERT_DIR/ca-cert.pem" \
    -CAkey "$CERT_DIR/ca-key.pem" \
    -out "$CERT_DIR/server-cert.pem" \
    -extensions v3_req \
    -extfile <(echo '[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names
[alt_names]
DNS.1 = localhost
DNS.2 = ipfs-accelerate.local
DNS.3 = *.ipfs-accelerate.local
IP.1 = 127.0.0.1
IP.2 = ::1')

# Set proper permissions
chmod 600 "$CERT_DIR"/*-key.pem
chmod 644 "$CERT_DIR"/*-cert.pem "$CERT_DIR"/*.csr

echo "Development certificates generated in $CERT_DIR"
EOF

    chmod +x deployments/ssl-scripts/generate_dev_certs.sh
    
    success "Certificate structure created"
}

# Setup Docker SSL configuration
setup_docker_ssl() {
    info "Setting up Docker SSL configuration..."
    
    # Update docker-compose.yml to include SSL
    if [ -f "deployments/docker-compose.yml" ]; then
        cat >> deployments/docker-compose.yml << 'EOF'

  # SSL/TLS Configuration
  nginx-ssl:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./certs:/etc/ssl/certs:ro
      - ./nginx-ssl.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      - web
    networks:
      - ipfs-accelerate
    restart: unless-stopped
    
  # Certificate management
  certbot:
    image: certbot/certbot
    volumes:
      - ./certs:/etc/letsencrypt
      - ./certbot-webroot:/var/www/certbot
    command: certonly --webroot --webroot-path=/var/www/certbot --email admin@ipfs-accelerate.com --agree-tos --no-eff-email -d ipfs-accelerate.local
    profiles:
      - letsencrypt
EOF
    fi
    
    success "Docker SSL configuration added"
}

# Setup Kubernetes SSL configuration  
setup_kubernetes_ssl() {
    info "Setting up Kubernetes SSL configuration..."
    
    # Create TLS secret manifest
    cat > deployments/kubernetes-tls.yaml << 'EOF'
apiVersion: v1
kind: Secret
metadata:
  name: ipfs-accelerate-tls
  namespace: default
type: kubernetes.io/tls
data:
  tls.crt: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0t  # Base64 encoded certificate
  tls.key: LS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0t  # Base64 encoded private key

---
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@ipfs-accelerate.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ipfs-accelerate-ingress
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - ipfs-accelerate.local
    secretName: ipfs-accelerate-tls
  rules:
  - host: ipfs-accelerate.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ipfs-accelerate-service
            port:
              number: 80
EOF

    success "Kubernetes TLS configuration created"
}

# Main execution
main() {
    info "Starting SSL/TLS setup for IPFS Accelerate Python..."
    
    case "$ENVIRONMENT" in
        "development")
            create_directories
            create_cert_structure
            deployments/ssl-scripts/generate_dev_certs.sh
            setup_docker_ssl
            ;;
        "staging"|"production")
            create_directories
            setup_letsencrypt
            setup_nginx_ssl
            setup_kubernetes_ssl
            ;;
        *)
            error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    success "SSL/TLS setup completed for $ENVIRONMENT environment"
    info "SSL configuration available at: deployments/ssl_config.yaml"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi