# Production-Grade Development Environment Deployment Guide

## Overview

This guide provides instructions for deploying a production-grade development environment for the Horse Racing Prediction application with high availability, redundancy, and comprehensive monitoring.

## Architecture

The production environment includes:

- **Application Layer**: Flask application with Gunicorn WSGI server
- **Load Balancer**: Nginx with SSL termination and rate limiting
- **Database**: PostgreSQL master-slave replication with automated failover
- **Caching**: Redis cluster with Sentinel for high availability
- **Monitoring**: Prometheus, Grafana, and alerting
- **Logging**: ELK stack (Elasticsearch, Logstash, Kibana)
- **Orchestration**: Docker Swarm for container management
- **Backup**: Automated backup and disaster recovery

## Prerequisites

### System Requirements

- Docker Engine 20.10+
- Docker Compose 2.0+
- Minimum 8GB RAM
- Minimum 50GB disk space
- Linux/macOS operating system

### Install Docker

```bash
# For Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# For macOS
brew install docker docker-compose

# For CentOS/RHEL
sudo yum install -y docker docker-compose
sudo systemctl enable docker
sudo systemctl start docker
```

## Deployment Steps

### 1. Initialize Docker Swarm

```bash
# Initialize Docker Swarm
docker swarm init

# Verify swarm status
docker node ls
```

### 2. Create Required Directories

```bash
# Create data directories
sudo mkdir -p /data/{postgres,redis,elasticsearch,grafana,prometheus}
sudo mkdir -p /backups/{postgres,logs}
sudo mkdir -p /var/log/{deployment,health-checks,backup}

# Set permissions
sudo chown -R $USER:$USER /data /backups /var/log
```

### 3. Configure Environment Variables

Create a `.env` file:

```bash
cat > .env << EOF
# Database Configuration
POSTGRES_DB=horse_racing_prediction
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_REPLICATION_USER=replicator
POSTGRES_REPLICATION_PASSWORD=replication_password_here

# Redis Configuration
REDIS_PASSWORD=redis_password_here

# Application Configuration
FLASK_ENV=production
SECRET_KEY=your_secret_key_here
API_KEY=your_api_key_here

# Monitoring Configuration
GRAFANA_ADMIN_PASSWORD=grafana_password_here

# Backup Configuration
BACKUP_RETENTION_DAYS=30
CLOUD_BACKUP_ENABLED=false
EOF
```

### 4. Deploy the Production Stack

```bash
# Build and deploy
./scripts/deployment/deploy-production.sh

# Or manually deploy
docker stack deploy -c docker-swarm-stack.yml horse-racing-prediction
```

### 5. Verify Deployment

```bash
# Check service status
docker service ls

# Run health checks
./scripts/health-checks/health-check.sh

# Check logs
docker service logs horse-racing-prediction_app
```

## Service Access

Once deployed, services are available at:

- **Application**: https://localhost (port 443)
- **Grafana**: https://localhost:3000
- **Kibana**: https://localhost:5601
- **Prometheus**: http://localhost:9090 (internal)

## Configuration Files

### Core Configuration

- `docker-swarm-stack.yml` - Main orchestration configuration
- `Dockerfile.production` - Production application container
- `nginx/nginx-production.conf` - Load balancer configuration

### Database Configuration

- `scripts/postgres/master-init.sh` - PostgreSQL master setup
- `scripts/postgres/slave-init.sh` - PostgreSQL slave setup
- `scripts/postgres/pg_hba.conf` - Authentication configuration

### Redis Configuration

- `scripts/redis/redis-master.conf` - Redis master configuration
- `scripts/redis/redis-slave.conf` - Redis slave configuration
- `scripts/redis/sentinel.conf` - Redis Sentinel configuration

### Monitoring Configuration

- `monitoring/prometheus/prometheus.yml` - Metrics collection
- `monitoring/prometheus/alert_rules.yml` - Alerting rules
- `monitoring/grafana/datasources/prometheus.yml` - Grafana datasources

### Logging Configuration

- `logging/elasticsearch/elasticsearch.yml` - Search engine configuration
- `logging/logstash/config/logstash.yml` - Log processing configuration
- `logging/logstash/pipeline/logstash.conf` - Log pipeline
- `logging/kibana/kibana.yml` - Log visualization configuration

## Operations

### Backup Operations

```bash
# Manual backup
./scripts/backup/backup-postgres.sh

# Automated backup (runs via cron)
0 2 * * * /path/to/scripts/backup/backup-postgres.sh
```

### Health Monitoring

```bash
# Single health check
./scripts/health-checks/health-check.sh

# Continuous monitoring
./scripts/health-checks/health-check.sh --continuous
```

### Scaling Services

```bash
# Scale application instances
docker service scale horse-racing-prediction_app=5

# Scale monitoring
docker service scale horse-racing-prediction_node-exporter=3
```

### Rolling Updates

```bash
# Update application
docker service update --image horse-racing-prediction-app:new-version horse-racing-prediction_app

# Update with zero downtime
docker service update --update-parallelism 1 --update-delay 30s horse-racing-prediction_app
```

### Rollback

```bash
# Rollback deployment
./scripts/deployment/deploy-production.sh rollback

# Manual rollback
docker service rollback horse-racing-prediction_app
```

## Monitoring and Alerting

### Prometheus Metrics

Key metrics monitored:

- Application response time and error rates
- Database connection pools and query performance
- Redis memory usage and connection counts
- System resources (CPU, memory, disk)
- Container health and resource usage

### Grafana Dashboards

Pre-configured dashboards for:

- Application performance overview
- Database monitoring
- Infrastructure monitoring
- Business metrics

### Alert Rules

Configured alerts for:

- Service downtime
- High error rates
- Resource exhaustion
- Database replication lag
- Security incidents

## Security

### SSL/TLS Configuration

- Nginx handles SSL termination
- Self-signed certificates for development
- Production should use Let's Encrypt or commercial certificates

### Network Security

- Services communicate on internal Docker networks
- Rate limiting on public endpoints
- Security headers configured in Nginx

### Access Control

- Database authentication required
- Redis password protection
- Monitoring dashboards require authentication

## Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   docker service logs horse-racing-prediction_<service>
   ```

2. **Database connection issues**
   ```bash
   docker exec -it $(docker ps -q -f name=postgres-master) psql -U postgres
   ```

3. **High memory usage**
   ```bash
   docker stats
   ```

4. **Log analysis**
   ```bash
   # View application logs in Kibana
   # Check Grafana dashboards for metrics
   ```

### Performance Tuning

1. **Database optimization**
   - Adjust PostgreSQL configuration in `scripts/postgres/`
   - Monitor slow queries in Grafana

2. **Application scaling**
   - Increase replica count based on load
   - Optimize Gunicorn worker configuration

3. **Caching optimization**
   - Monitor Redis memory usage
   - Adjust cache TTL values

## Maintenance

### Regular Tasks

1. **Daily**
   - Monitor service health
   - Check backup completion
   - Review error logs

2. **Weekly**
   - Update security patches
   - Review performance metrics
   - Clean up old logs

3. **Monthly**
   - Update application dependencies
   - Review and update monitoring rules
   - Test disaster recovery procedures

### Disaster Recovery

1. **Database Recovery**
   ```bash
   # Restore from backup
   gunzip -c /backups/postgres/postgres_backup_YYYYMMDD_HHMMSS.sql.gz | \
   docker exec -i $(docker ps -q -f name=postgres-master) psql -U postgres
   ```

2. **Full Environment Recovery**
   ```bash
   # Redeploy stack
   docker stack rm horse-racing-prediction
   ./scripts/deployment/deploy-production.sh
   ```

## Support

For issues and questions:

1. Check service logs: `docker service logs <service_name>`
2. Run health checks: `./scripts/health-checks/health-check.sh`
3. Review monitoring dashboards in Grafana
4. Check application logs in Kibana

## Next Steps

1. Configure external load balancer for true high availability
2. Set up multi-node Docker Swarm cluster
3. Implement external backup storage (AWS S3, Google Cloud Storage)
4. Configure external monitoring and alerting (PagerDuty, Slack)
5. Set up CI/CD pipeline for automated deployments