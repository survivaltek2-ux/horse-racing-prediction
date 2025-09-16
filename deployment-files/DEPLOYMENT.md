# Horse Racing Prediction System - Deployment Guide

This guide provides instructions for deploying the Horse Racing Prediction System to a production server.

## Quick Start

The easiest way to deploy this application is using the automated deployment script:

```bash
# Make the script executable (if not already)
chmod +x deploy.sh

# Run the deployment script
./deploy.sh
```

## Prerequisites

### System Requirements
- **Operating System**: Ubuntu 18.04+, Debian 10+, CentOS 7+, or RHEL 7+
- **RAM**: Minimum 2GB, Recommended 4GB+
- **Storage**: Minimum 10GB free space
- **Network**: Internet connection for downloading dependencies

### User Requirements
- Non-root user with sudo privileges
- SSH access to the server

## What the Deployment Script Does

The `deploy.sh` script automatically:

1. **System Setup**
   - Detects your operating system
   - Installs system dependencies (Python, Nginx, Supervisor, etc.)
   - Creates a dedicated application user (`hrp`)

2. **Application Setup**
   - Creates application directory (`/opt/horse-racing-prediction`)
   - Sets up Python virtual environment
   - Installs all Python dependencies from `requirements.txt`
   - Configures proper file permissions

3. **Database Setup**
   - Initializes SQLite database
   - Creates all necessary tables
   - Sets up data directory with proper permissions

4. **Web Server Configuration**
   - Configures Nginx as reverse proxy
   - Sets up Gunicorn as WSGI server
   - Configures SSL-ready virtual host

5. **Process Management**
   - Sets up Supervisor for process management
   - Creates systemd service as backup
   - Configures automatic restart on failure

6. **Security & Monitoring**
   - Configures firewall (UFW/firewalld)
   - Sets up log rotation
   - Creates automated backup script
   - Generates secure environment configuration

## Manual Deployment Steps

If you prefer to deploy manually or need to customize the process:

### 1. System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv python3-dev \
    build-essential nginx supervisor git sqlite3 libsqlite3-dev \
    pkg-config libssl-dev libffi-dev libjpeg-dev libpng-dev \
    libfreetype6-dev libblas-dev liblapack-dev gfortran libatlas-base-dev
```

**CentOS/RHEL:**
```bash
sudo yum update -y
sudo yum install -y python3 python3-pip python3-devel gcc gcc-c++ make \
    nginx supervisor git sqlite sqlite-devel openssl-devel libffi-devel \
    libjpeg-turbo-devel libpng-devel freetype-devel blas-devel lapack-devel
```

### 2. Application User
```bash
sudo useradd --system --shell /bin/bash --home-dir /opt/horse-racing-prediction --create-home hrp
```

### 3. Application Files
```bash
sudo cp -r . /opt/horse-racing-prediction/
sudo chown -R hrp:hrp /opt/horse-racing-prediction
```

### 4. Python Environment
```bash
sudo -u hrp python3 -m venv /opt/horse-racing-prediction/venv
sudo -u hrp /opt/horse-racing-prediction/venv/bin/pip install -r /opt/horse-racing-prediction/requirements.txt
```

### 5. Database Initialization
```bash
cd /opt/horse-racing-prediction
sudo -u hrp ./venv/bin/python -c "
from app import app, db
with app.app_context():
    db.create_all()
    print('Database initialized')
"
```

## Configuration

### Environment Variables

Edit `/opt/horse-racing-prediction/.env`:

```bash
# Production settings
FLASK_ENV=production
SECRET_KEY=your-secure-secret-key-here
DEBUG=False

# API Keys (add your actual keys)
SAMPLE_API_KEY=your_sample_api_key_here
ODDS_API_API_KEY=your_odds_api_key_here
RAPID_API_API_KEY=your_rapidapi_key_here
THERACINGAPI_USERNAME=your_username_here
THERACINGAPI_PASSWORD=your_password_here

# Database
DATABASE_URL=sqlite:///opt/horse-racing-prediction/data/hrp_database.db
```

### Domain Configuration

Update the domain name in the deployment script or manually edit Nginx configuration:

```bash
# Set your domain before running the script
export DOMAIN_NAME=yourdomain.com
./deploy.sh
```

Or edit `/etc/nginx/sites-available/horse-racing-prediction` after deployment.

## SSL/HTTPS Setup

To enable HTTPS with Let's Encrypt:

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx  # Ubuntu/Debian
sudo yum install certbot python3-certbot-nginx      # CentOS/RHEL

# Get SSL certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal (already set up by certbot)
sudo crontab -l | grep certbot
```

## Service Management

### Start/Stop Services
```bash
# Application
sudo supervisorctl start hrp-app
sudo supervisorctl stop hrp-app
sudo supervisorctl restart hrp-app

# Nginx
sudo systemctl start nginx
sudo systemctl stop nginx
sudo systemctl restart nginx
```

### View Logs
```bash
# Application logs
sudo tail -f /opt/horse-racing-prediction/logs/supervisor.log
sudo tail -f /opt/horse-racing-prediction/logs/gunicorn_error.log

# Nginx logs
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log
```

### Check Status
```bash
# Application status
sudo supervisorctl status hrp-app

# Nginx status
sudo systemctl status nginx

# Check if application is responding
curl http://localhost:8000/
```

## Backup and Maintenance

### Manual Backup
```bash
sudo -u hrp /opt/horse-racing-prediction/backup.sh
```

### Database Backup
```bash
# Create database backup
sudo -u hrp sqlite3 /opt/horse-racing-prediction/data/hrp_database.db ".backup /opt/horse-racing-prediction/backups/db_backup_$(date +%Y%m%d).db"
```

### Update Application
```bash
# Stop application
sudo supervisorctl stop hrp-app

# Update code
cd /opt/horse-racing-prediction
sudo -u hrp git pull origin main

# Install new dependencies (if any)
sudo -u hrp ./venv/bin/pip install -r requirements.txt

# Run database migrations (if any)
sudo -u hrp ./venv/bin/python migrate_enhanced_schema.py

# Start application
sudo supervisorctl start hrp-app
```

## Troubleshooting

### Common Issues

1. **Application won't start**
   ```bash
   # Check logs
   sudo tail -f /opt/horse-racing-prediction/logs/supervisor.log
   
   # Check Python dependencies
   sudo -u hrp /opt/horse-racing-prediction/venv/bin/pip list
   ```

2. **Database errors**
   ```bash
   # Check database file permissions
   ls -la /opt/horse-racing-prediction/data/
   
   # Recreate database
   sudo -u hrp rm /opt/horse-racing-prediction/data/hrp_database.db
   cd /opt/horse-racing-prediction
   sudo -u hrp ./venv/bin/python -c "from app import app, db; app.app_context().push(); db.create_all()"
   ```

3. **Nginx errors**
   ```bash
   # Test Nginx configuration
   sudo nginx -t
   
   # Check if port is in use
   sudo netstat -tlnp | grep :80
   ```

4. **Permission errors**
   ```bash
   # Fix ownership
   sudo chown -R hrp:hrp /opt/horse-racing-prediction
   
   # Fix permissions
   sudo chmod -R 755 /opt/horse-racing-prediction
   sudo chmod 600 /opt/horse-racing-prediction/.env
   ```

### Performance Tuning

1. **Gunicorn Workers**
   Edit `/opt/horse-racing-prediction/gunicorn.conf.py`:
   ```python
   workers = multiprocessing.cpu_count() * 2 + 1  # Adjust based on your server
   ```

2. **Nginx Caching**
   Add to Nginx configuration:
   ```nginx
   location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
       expires 1y;
       add_header Cache-Control "public, immutable";
   }
   ```

## Security Considerations

1. **Firewall**: The script configures basic firewall rules
2. **User Permissions**: Application runs as non-root user
3. **File Permissions**: Sensitive files have restricted access
4. **Environment Variables**: Secrets stored in protected .env file
5. **Database**: SQLite file has restricted permissions

### Additional Security Steps

1. **Change default SSH port**
2. **Set up fail2ban for intrusion prevention**
3. **Regular security updates**
4. **Monitor logs for suspicious activity**
5. **Use strong passwords for API keys**

## Monitoring

### Health Checks
```bash
# Application health
curl http://localhost:8000/health

# Database health
sudo -u hrp sqlite3 /opt/horse-racing-prediction/data/hrp_database.db "SELECT COUNT(*) FROM users;"
```

### Log Monitoring
```bash
# Real-time log monitoring
sudo tail -f /opt/horse-racing-prediction/logs/supervisor.log | grep ERROR
```

## Support

If you encounter issues:

1. Check the logs in `/opt/horse-racing-prediction/logs/`
2. Verify all services are running
3. Check firewall and network connectivity
4. Ensure all dependencies are installed
5. Verify file permissions and ownership

For additional help, review the application documentation or check the GitHub repository for known issues and solutions.