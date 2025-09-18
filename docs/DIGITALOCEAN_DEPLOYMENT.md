# DigitalOcean Deployment Guide

This guide provides comprehensive instructions for deploying the Horse Racing Prediction application to DigitalOcean using the automated deployment script.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Configuration](#configuration)
3. [Deployment Process](#deployment-process)
4. [Post-Deployment](#post-deployment)
5. [Monitoring and Maintenance](#monitoring-and-maintenance)
6. [Troubleshooting](#troubleshooting)
7. [Security Considerations](#security-considerations)

## Prerequisites

### Required Tools

Before running the deployment script, ensure you have the following tools installed:

```bash
# Install DigitalOcean CLI
brew install doctl

# Install Docker (if not already installed)
brew install docker

# Install other required tools
brew install curl jq openssh
```

### DigitalOcean Setup

1. **Create a DigitalOcean Account**
   - Sign up at [digitalocean.com](https://digitalocean.com)
   - Add a payment method

2. **Generate API Token**
   - Go to API → Tokens/Keys
   - Generate a new Personal Access Token
   - Copy the token (you'll need it for configuration)

3. **Add SSH Key**
   - Generate SSH key if you don't have one:
     ```bash
     ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
     ```
   - Add your public key to DigitalOcean:
     ```bash
     doctl compute ssh-key import my-key --public-key-file ~/.ssh/id_rsa.pub
     ```

4. **Configure doctl**
   ```bash
   doctl auth init
   # Enter your API token when prompted
   ```

## Configuration

### Environment Configuration

1. **Copy the configuration template:**
   ```bash
   cp .env.digitalocean.example .env.digitalocean
   ```

2. **Edit the configuration file:**
   ```bash
   nano .env.digitalocean
   ```

### Required Configuration Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DO_API_TOKEN` | DigitalOcean API token | `dop_v1_abc123...` |
| `DO_SSH_KEY_NAME` | Name of SSH key in DO | `my-deployment-key` |
| `DO_REGION` | Deployment region | `nyc3` |
| `DO_SIZE` | Droplet size | `s-2vcpu-2gb` |
| `DO_DOMAIN` | Your domain name | `myapp.com` |
| `SSL_EMAIL` | Email for SSL certificates | `admin@myapp.com` |

### Optional Configuration Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_PORT` | `8000` | Application port |
| `DB_NAME` | `horse_racing_db` | Database name |
| `DB_USER` | `app_user` | Database user |
| `ENABLE_SSL` | `true` | Enable SSL certificates |
| `ENABLE_MONITORING` | `true` | Enable monitoring stack |
| `ENABLE_BACKUPS` | `true` | Enable automated backups |

## Deployment Process

### Step 1: Validate Configuration

Before deployment, validate your configuration:

```bash
# Check DigitalOcean authentication
doctl account get

# Verify SSH key exists
doctl compute ssh-key list

# Test SSH key access
ssh-add ~/.ssh/id_rsa
```

### Step 2: Run Deployment Script

Execute the deployment script:

```bash
./scripts/deployment/deploy-digitalocean.sh
```

The script will:

1. **Validate Prerequisites** - Check required tools and configuration
2. **Create Droplet** - Provision a new DigitalOcean droplet
3. **Configure Security** - Set up firewall, SSH hardening, and fail2ban
4. **Install Dependencies** - Install Docker, monitoring tools, and SSL tools
5. **Deploy Application** - Transfer code, build containers, and start services
6. **Setup Monitoring** - Deploy Prometheus, Grafana, and health checks
7. **Verify Deployment** - Run health checks and display summary

### Step 3: Monitor Deployment

The script provides real-time logging. Watch for:

- ✅ Green success messages
- ⚠️ Yellow warning messages
- ❌ Red error messages

### Step 4: Deployment Summary

Upon successful completion, you'll see:

```
=== DEPLOYMENT SUMMARY ===
Droplet ID: 123456789
Droplet IP: 192.168.1.100
Application URL: http://192.168.1.100:8000
Domain URL: https://myapp.com
Monitoring URLs:
  - Grafana: http://192.168.1.100:3000
  - Prometheus: http://192.168.1.100:9090
SSH Access: ssh appuser@192.168.1.100
```

## Post-Deployment

### DNS Configuration

Update your domain's DNS records:

```
Type: A
Name: @
Value: [DROPLET_IP]
TTL: 300

Type: A
Name: www
Value: [DROPLET_IP]
TTL: 300
```

### SSL Certificate Verification

Check SSL certificate status:

```bash
# SSH into the droplet
ssh appuser@[DROPLET_IP]

# Check certificate
sudo certbot certificates

# Test SSL renewal
sudo certbot renew --dry-run
```

### Application Verification

1. **Health Check:**
   ```bash
   curl -f https://yourdomain.com/health
   ```

2. **Application Access:**
   - Open https://yourdomain.com in your browser
   - Verify all functionality works correctly

3. **Monitoring Access:**
   - Grafana: http://[DROPLET_IP]:3000
   - Prometheus: http://[DROPLET_IP]:9090

## Monitoring and Maintenance

### Log Monitoring

```bash
# SSH into droplet
ssh appuser@[DROPLET_IP]

# View application logs
cd /opt/horse-racing-prediction
docker-compose logs -f app

# View system logs
sudo journalctl -f

# View deployment logs
tail -f /var/log/deployment.log
```

### Health Checks

The deployment includes automated health checks:

- **Application Health:** Every 5 minutes
- **Database Health:** Every 5 minutes
- **Redis Health:** Every 5 minutes

View health check logs:
```bash
sudo tail -f /var/log/health-check.log
```

### Backup Management

Automated backups run daily at 2 AM:

```bash
# View backup logs
sudo tail -f /var/log/backup.log

# Manual backup
sudo /opt/backups/backup.sh

# List backups
ls -la /opt/backups/
```

### Monitoring Stack

#### Grafana Dashboard

1. Access: http://[DROPLET_IP]:3000
2. Login: admin / [GRAFANA_PASSWORD]
3. Import dashboards for application metrics

#### Prometheus Metrics

1. Access: http://[DROPLET_IP]:9090
2. Query application metrics
3. Set up alerting rules

### Updates and Maintenance

#### Application Updates

```bash
# SSH into droplet
ssh appuser@[DROPLET_IP]

# Pull latest code (if using git)
cd /opt/horse-racing-prediction
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up -d --build
```

#### System Updates

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Docker images
cd /opt/horse-racing-prediction
docker-compose pull
docker-compose up -d
```

## Troubleshooting

### Common Issues

#### 1. Deployment Fails During Droplet Creation

**Error:** `Error creating droplet: insufficient quota`

**Solution:**
- Check your DigitalOcean account limits
- Try a smaller droplet size
- Contact DigitalOcean support

#### 2. SSH Connection Fails

**Error:** `Permission denied (publickey)`

**Solution:**
```bash
# Verify SSH key is added to ssh-agent
ssh-add ~/.ssh/id_rsa

# Check if key exists in DigitalOcean
doctl compute ssh-key list

# Re-add key if necessary
doctl compute ssh-key import my-key --public-key-file ~/.ssh/id_rsa.pub
```

#### 3. SSL Certificate Generation Fails

**Error:** `Failed to obtain certificate`

**Solution:**
- Ensure DNS records point to the droplet
- Check domain ownership
- Verify port 80 is accessible

#### 4. Application Won't Start

**Error:** `Service app is not running`

**Solution:**
```bash
# SSH into droplet
ssh appuser@[DROPLET_IP]

# Check container logs
cd /opt/horse-racing-prediction
docker-compose logs app

# Check container status
docker-compose ps

# Restart services
docker-compose restart
```

### Log Locations

| Component | Log Location |
|-----------|--------------|
| Deployment | `/var/log/deployment.log` |
| Application | `docker-compose logs app` |
| Nginx | `/var/log/nginx/` |
| System | `/var/log/syslog` |
| Health Checks | `/var/log/health-check.log` |
| Backups | `/var/log/backup.log` |

### Emergency Procedures

#### Rollback Deployment

```bash
# SSH into droplet
ssh appuser@[DROPLET_IP]

# Stop services
cd /opt/horse-racing-prediction
docker-compose down

# Restore from backup
sudo /opt/backups/restore.sh [BACKUP_DATE]

# Restart services
docker-compose up -d
```

#### Complete Cleanup

```bash
# Destroy droplet (WARNING: This deletes everything)
doctl compute droplet delete [DROPLET_ID]

# Remove local deployment info
rm deployment-info.txt
```

## Security Considerations

### Firewall Configuration

The deployment automatically configures UFW with these rules:

- SSH (22): Limited to your IP
- HTTP (80): Open (redirects to HTTPS)
- HTTPS (443): Open
- Application (8000): Restricted to localhost
- Monitoring (3000, 9090): Restricted to your IP

### SSH Hardening

Automatic SSH security measures:

- Disable root login
- Disable password authentication
- Change default SSH port (optional)
- Install fail2ban for intrusion prevention

### SSL/TLS Configuration

- Automatic Let's Encrypt certificates
- Strong cipher suites
- HSTS headers
- Certificate auto-renewal

### Database Security

- Strong random passwords
- Network isolation
- Regular backups
- Connection encryption

### Application Security

- Environment-specific configuration
- Secret key generation
- Rate limiting
- Security headers

### Monitoring Security

- Secure monitoring endpoints
- Authentication required
- Log monitoring for security events

## Advanced Configuration

### Custom Domain Setup

For multiple domains or subdomains:

```bash
# Edit Nginx configuration
sudo nano /opt/horse-racing-prediction/nginx.conf

# Add additional server blocks
# Restart Nginx
docker-compose restart nginx
```

### Scaling Considerations

For high-traffic applications:

1. **Vertical Scaling:** Upgrade droplet size
2. **Horizontal Scaling:** Use load balancers
3. **Database Scaling:** Consider managed databases
4. **CDN Integration:** Use DigitalOcean Spaces

### Backup Strategies

#### Off-site Backups

Configure DigitalOcean Spaces for backup storage:

```bash
# Install s3cmd
sudo apt install s3cmd

# Configure for DigitalOcean Spaces
s3cmd --configure

# Modify backup script to upload to Spaces
```

#### Database Replication

For critical applications, consider:

- DigitalOcean Managed Databases
- Read replicas
- Point-in-time recovery

## Support and Resources

### Documentation

- [DigitalOcean Documentation](https://docs.digitalocean.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)

### Community

- [DigitalOcean Community](https://www.digitalocean.com/community)
- [Docker Community](https://www.docker.com/community)

### Professional Support

For production deployments, consider:

- DigitalOcean Professional Services
- Third-party DevOps consultants
- Managed hosting solutions

---

**Note:** This deployment script is designed for production use but should be thoroughly tested in a staging environment before deploying critical applications.