#!/bin/bash

# SSL Certificate Setup Script
# Horse Racing Prediction System

set -e

echo "🔐 Setting up SSL certificates for production deployment..."

# Create SSL directory
SSL_DIR="/etc/nginx/ssl"
LOCAL_SSL_DIR="./ssl"

# Create local SSL directory for development
mkdir -p "$LOCAL_SSL_DIR"

# Function to generate self-signed certificates for development/testing
generate_self_signed_cert() {
    echo "📋 Generating self-signed SSL certificate for development/testing..."
    
    # Generate private key
    openssl genrsa -out "$LOCAL_SSL_DIR/key.pem" 2048
    
    # Generate certificate signing request
    openssl req -new -key "$LOCAL_SSL_DIR/key.pem" -out "$LOCAL_SSL_DIR/cert.csr" -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=localhost"
    
    # Generate self-signed certificate
    openssl x509 -req -days 365 -in "$LOCAL_SSL_DIR/cert.csr" -signkey "$LOCAL_SSL_DIR/key.pem" -out "$LOCAL_SSL_DIR/cert.pem"
    
    # Generate DH parameters
    openssl dhparam -out "$LOCAL_SSL_DIR/dhparam.pem" 2048
    
    # Set proper permissions
    chmod 600 "$LOCAL_SSL_DIR/key.pem"
    chmod 644 "$LOCAL_SSL_DIR/cert.pem"
    chmod 644 "$LOCAL_SSL_DIR/dhparam.pem"
    
    echo "✅ Self-signed SSL certificate generated successfully!"
    echo "   Certificate: $LOCAL_SSL_DIR/cert.pem"
    echo "   Private Key: $LOCAL_SSL_DIR/key.pem"
    echo "   DH Params: $LOCAL_SSL_DIR/dhparam.pem"
}

# Function to setup Let's Encrypt certificates (for production)
setup_letsencrypt() {
    echo "📋 Setting up Let's Encrypt SSL certificates..."
    echo "⚠️  This requires a valid domain name and internet access"
    
    # Check if certbot is installed
    if ! command -v certbot &> /dev/null; then
        echo "Installing certbot..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install certbot
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get update
            sudo apt-get install -y certbot python3-certbot-nginx
        fi
    fi
    
    # Domain name (should be set as environment variable)
    DOMAIN_NAME=${DOMAIN_NAME:-"your-domain.com"}
    
    echo "🌐 Setting up SSL for domain: $DOMAIN_NAME"
    echo "⚠️  Make sure your domain points to this server's IP address"
    
    # Generate Let's Encrypt certificate
    # Note: This is a dry-run command for safety
    echo "Run this command when ready for production:"
    echo "sudo certbot --nginx -d $DOMAIN_NAME --non-interactive --agree-tos --email your-email@domain.com"
}

# Function to copy certificates to Docker volume
setup_docker_ssl() {
    echo "📋 Setting up SSL certificates for Docker deployment..."
    
    # Create Docker volume for SSL certificates
    docker volume create ssl-certs || true
    
    # Copy certificates to Docker volume (using a temporary container)
    docker run --rm -v ssl-certs:/ssl -v "$(pwd)/$LOCAL_SSL_DIR":/source alpine sh -c "
        cp /source/cert.pem /ssl/
        cp /source/key.pem /ssl/
        cp /source/dhparam.pem /ssl/
        chmod 644 /ssl/cert.pem
        chmod 600 /ssl/key.pem
        chmod 644 /ssl/dhparam.pem
    "
    
    echo "✅ SSL certificates copied to Docker volume 'ssl-certs'"
}

# Main execution
case "${1:-self-signed}" in
    "self-signed")
        generate_self_signed_cert
        setup_docker_ssl
        ;;
    "letsencrypt")
        setup_letsencrypt
        ;;
    "docker")
        setup_docker_ssl
        ;;
    *)
        echo "Usage: $0 [self-signed|letsencrypt|docker]"
        echo "  self-signed: Generate self-signed certificates for development"
        echo "  letsencrypt: Setup Let's Encrypt certificates for production"
        echo "  docker: Copy existing certificates to Docker volume"
        exit 1
        ;;
esac

echo ""
echo "🔐 SSL Certificate Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Update your .env.production file with correct SSL paths"
echo "2. Ensure your Nginx configuration uses the SSL certificates"
echo "3. Test the SSL configuration before deployment"
echo ""
echo "For production deployment:"
echo "- Use a valid domain name"
echo "- Setup DNS to point to your server"
echo "- Use Let's Encrypt for free SSL certificates"
echo "- Consider using a CDN for better performance"