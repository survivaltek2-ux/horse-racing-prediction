# Automated Deployment Scripts

This directory contains a comprehensive set of automated deployment scripts for the Horse Racing Prediction application. These scripts provide a complete deployment solution with monitoring, backup, SSL management, and rollback capabilities.

## 📁 Script Overview

| Script | Purpose | Description |
|--------|---------|-------------|
| `auto-deploy.sh` | **Main Deployment** | Orchestrates the entire deployment process |
| `setup-environment.sh` | **Environment Setup** | Configures system environment and dependencies |
| `setup-database.sh` | **Database Management** | Handles database setup, migrations, and optimization |
| `ssl-setup.sh` | **SSL/TLS Management** | Automates SSL certificate creation and management |
| `monitoring.sh` | **Health Monitoring** | Provides comprehensive system and application monitoring |
| `backup-rollback.sh` | **Backup & Recovery** | Handles backups, rollbacks, and disaster recovery |
| `test-deployment.sh` | **Testing & Validation** | Tests and validates all deployment scripts |

## 🚀 Quick Start

### 1. Initial Deployment

```bash
# Full production deployment
./scripts/deployment/auto-deploy.sh production deploy

# Development deployment
./scripts/deployment/auto-deploy.sh development deploy

# Staging deployment with SSL
./scripts/deployment/auto-deploy.sh staging deploy --ssl --domain your-domain.com
```

### 2. Test Before Deployment

```bash
# Run comprehensive tests
./scripts/deployment/test-deployment.sh development

# Quick validation tests
./scripts/deployment/test-deployment.sh development --quick

# Skip destructive tests
./scripts/deployment/test-deployment.sh production --skip-destructive
```

## 📋 Detailed Usage

### Main Deployment Script (`auto-deploy.sh`)

The main orchestration script that handles the complete deployment process.

```bash
# Basic usage
./auto-deploy.sh [environment] [action] [options]

# Examples
./auto-deploy.sh production deploy --ssl --domain example.com
./auto-deploy.sh development deploy --skip-ssl
./auto-deploy.sh staging rollback --version v1.2.3
./auto-deploy.sh production status
```

**Environments:** `development`, `staging`, `production`

**Actions:**
- `deploy` - Full deployment
- `update` - Update existing deployment
- `rollback` - Rollback to previous version
- `status` - Check deployment status
- `health` - Run health checks

**Options:**
- `--ssl` - Enable SSL setup
- `--domain DOMAIN` - Specify domain for SSL
- `--skip-ssl` - Skip SSL configuration
- `--skip-backup` - Skip backup creation
- `--dry-run` - Show what would be done without executing
- `--force` - Force deployment even if checks fail
- `--version VERSION` - Deploy specific version

### Environment Setup (`setup-environment.sh`)

Configures the system environment and installs dependencies.

```bash
# Setup development environment
./setup-environment.sh development setup

# Create environment file only
./setup-environment.sh production create-env

# Install system dependencies
./setup-environment.sh production install-deps

# Setup Docker environment
./setup-environment.sh staging setup-docker
```

### Database Management (`setup-database.sh`)

Handles all database-related operations.

```bash
# Setup new database
./setup-database.sh production setup

# Run migrations
./setup-database.sh production migrate

# Import data
./setup-database.sh development import --file data.sql

# Create backup
./setup-database.sh production backup --name daily_backup

# Optimize database
./setup-database.sh production optimize
```

### SSL Management (`ssl-setup.sh`)

Automates SSL certificate management with Let's Encrypt.

```bash
# Setup SSL with Let's Encrypt
./ssl-setup.sh production setup --domain your-domain.com

# Create self-signed certificate (development)
./ssl-setup.sh development setup --domain localhost

# Renew certificates
./ssl-setup.sh production renew

# Check certificate status
./ssl-setup.sh production status --domain your-domain.com
```

### Monitoring (`monitoring.sh`)

Comprehensive monitoring and health checking.

```bash
# Run all health checks
./monitoring.sh production health

# Check specific service
./monitoring.sh production check --service app

# Setup monitoring stack
./monitoring.sh production setup-monitoring

# Generate performance report
./monitoring.sh production performance --duration 24h
```

### Backup & Recovery (`backup-rollback.sh`)

Handles backups, rollbacks, and disaster recovery.

```bash
# Create full backup
./backup-rollback.sh production backup --type full

# Database backup only
./backup-rollback.sh production backup --type database

# Rollback database
./backup-rollback.sh production rollback --type database --backup backup_20231201

# Setup scheduled backups
./backup-rollback.sh production setup-schedule
```

## 🔧 Configuration

### Environment Variables

Each environment requires specific configuration. Create or update these files:

- `.env.development` - Development configuration
- `.env.staging` - Staging configuration  
- `.env.production` - Production configuration

**Required Variables:**
```bash
# Application
FLASK_ENV=production
FLASK_APP=app.py
SECRET_KEY=your-secret-key

# Database
POSTGRES_USER=app_user
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=horse_racing_db
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Redis
REDIS_URL=redis://redis:6379/0

# SSL (for production)
SSL_DOMAIN=your-domain.com
SSL_EMAIL=admin@your-domain.com

# Monitoring
ENABLE_MONITORING=true
GRAFANA_ADMIN_PASSWORD=admin_password
```

### Server Requirements

**Minimum Requirements:**
- Ubuntu 20.04+ or CentOS 8+
- 2 CPU cores
- 4GB RAM
- 20GB disk space
- Docker and Docker Compose

**Recommended for Production:**
- 4+ CPU cores
- 8GB+ RAM
- 50GB+ SSD storage
- Load balancer (if scaling)

## 🔒 Security Features

### SSL/TLS
- Automatic Let's Encrypt certificate generation
- Certificate auto-renewal
- Strong cipher suites
- HSTS headers

### Database Security
- Encrypted connections
- User privilege separation
- Regular security updates
- Backup encryption

### Application Security
- Environment variable isolation
- Secret management
- Security headers
- Input validation

## 📊 Monitoring & Alerting

### Health Checks
- Application health endpoints
- Database connectivity
- Redis connectivity
- SSL certificate expiry
- Disk space monitoring
- Memory usage monitoring

### Metrics Collection
- Prometheus metrics
- Grafana dashboards
- Custom application metrics
- Performance monitoring

### Alerting
- Email notifications
- Slack integration
- Critical error alerts
- Performance degradation alerts

## 🔄 Backup Strategy

### Automated Backups
- Daily database backups
- Weekly full application backups
- Monthly archive backups
- Configurable retention policies

### Backup Storage
- Local backup storage
- Remote backup upload (S3, etc.)
- Backup verification
- Encrypted backup files

### Recovery Procedures
- Point-in-time recovery
- Full system restore
- Database rollback
- Application rollback

## 🚨 Troubleshooting

### Common Issues

**1. Permission Denied Errors**
```bash
# Fix script permissions
chmod +x scripts/deployment/*.sh

# Fix Docker permissions
sudo usermod -aG docker $USER
```

**2. Database Connection Issues**
```bash
# Check database status
./monitoring.sh development check --service postgres

# Reset database
./setup-database.sh development reset
```

**3. SSL Certificate Issues**
```bash
# Check certificate status
./ssl-setup.sh production status --domain your-domain.com

# Force certificate renewal
./ssl-setup.sh production renew --force
```

**4. Docker Issues**
```bash
# Restart Docker services
docker-compose down && docker-compose up -d

# Clean Docker system
docker system prune -f
```

### Log Locations

- Application logs: `logs/app/`
- Deployment logs: `logs/deployment/`
- Database logs: `logs/database/`
- Nginx logs: `logs/nginx/`
- Monitoring logs: `logs/monitoring/`

### Getting Help

1. Check the logs in the appropriate directory
2. Run the test script to validate configuration
3. Use the `--dry-run` option to see what would be executed
4. Check the monitoring dashboard for system status

## 🔄 Update Process

### Regular Updates

```bash
# Update application code
git pull origin main

# Update deployment
./auto-deploy.sh production update

# Update dependencies
./setup-environment.sh production update-deps
```

### Security Updates

```bash
# Update system packages
./setup-environment.sh production security-update

# Update SSL certificates
./ssl-setup.sh production renew

# Update Docker images
docker-compose pull && docker-compose up -d
```

## 📈 Scaling

### Horizontal Scaling
- Load balancer configuration
- Database read replicas
- Redis clustering
- Container orchestration

### Vertical Scaling
- Resource monitoring
- Performance optimization
- Database tuning
- Cache optimization

## 🤝 Contributing

When modifying deployment scripts:

1. Test changes in development environment
2. Run the test suite: `./test-deployment.sh development`
3. Update documentation if needed
4. Follow security best practices
5. Test rollback procedures

## 📄 License

This deployment system is part of the Horse Racing Prediction application. See the main project license for details.

---

**Note:** Always test deployment scripts in a development environment before using in production. Keep backups of your data and configuration files.