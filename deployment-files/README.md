# Horse Racing Prediction - Deployment Files

This folder contains all the files needed to deploy your Horse Racing Prediction application to a production server.

## 📁 Files Included

### 🚀 Main Deployment
- **`deploy.sh`** - Main deployment script that automatically sets up everything
  - Installs system dependencies (Python, Nginx, SQLite, etc.)
  - Creates application user and directories
  - Sets up Python virtual environment
  - Configures database and web server
  - Handles security and monitoring setup

### 📖 Documentation
- **`DEPLOYMENT.md`** - Complete deployment guide
  - Step-by-step instructions
  - Configuration options
  - Troubleshooting guide
  - Security considerations

### 🐳 Docker Deployment (Alternative)
- **`Dockerfile`** - Container configuration for the application
- **`docker-compose.yml`** - Multi-container setup with Nginx
- **`nginx.conf`** - Production-ready Nginx configuration

## 🎯 Quick Start

### Option 1: Automated Deployment (Recommended)
```bash
# Copy files to your server
scp -r deployment-files/ user@your-server:/path/to/deployment/

# On your server, run:
chmod +x deploy.sh
./deploy.sh
```

### Option 2: Docker Deployment
```bash
# Copy files to your server
scp -r deployment-files/ user@your-server:/path/to/deployment/

# On your server, run:
docker-compose up -d
```

## 📋 Prerequisites

Before running the deployment:
1. **Server Requirements**: Ubuntu 18.04+, CentOS 7+, or similar Linux distribution
2. **Root Access**: You'll need sudo privileges
3. **Domain/IP**: Have your server's IP address or domain name ready
4. **Environment Variables**: Copy `.env.example` to `.env` and configure your settings

## 🔧 What Gets Deployed

- **Web Application**: Flask app running on Gunicorn
- **Web Server**: Nginx reverse proxy with security headers
- **Database**: SQLite with proper initialization
- **Security**: Firewall, SSL-ready, secure permissions
- **Monitoring**: Log rotation, health checks, automated backups

## 📞 Support

For detailed instructions, see `DEPLOYMENT.md` in this folder.

---
*Generated for Horse Racing Prediction Application*