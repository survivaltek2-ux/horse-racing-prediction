# High Availability Deployment Guide

This guide provides step-by-step instructions for deploying the Horse Racing Prediction application with full high availability, redundancy, and disaster recovery capabilities.

## Architecture Overview

The high-availability setup includes:

- **Load Balancer**: HAProxy with health checks and SSL termination
- **Application Layer**: 6 redundant application instances across 3 zones
- **Database Layer**: PostgreSQL primary-replica setup with automatic failover
- **Cache Layer**: Redis cluster with Sentinel for automatic failover
- **Storage Layer**: Distributed storage with GlusterFS and MinIO
- **Monitoring**: Prometheus, Grafana, and custom alerting
- **Backup System**: Automated backups with retention and S3 integration
- **Failover Automation**: Comprehensive failover scripts for all services

## Prerequisites

### Infrastructure Requirements

- **Minimum 3 nodes** for multi-zone deployment
- **8GB RAM per node** (minimum)
- **100GB storage per node** (minimum)
- **Docker Swarm cluster** initialized
- **Network connectivity** between all nodes

### Software Requirements

- Docker Engine 20.10+
- Docker Compose 2.0+
- Docker Swarm mode enabled
- SSL certificates (for production)

## Initial Setup

### 1. Prepare Docker Swarm Cluster

```bash
# On manager node
docker swarm init --advertise-addr <MANAGER_IP>

# On worker nodes
docker swarm join --token <TOKEN> <MANAGER_IP>:2377

# Label nodes for zone placement
docker node update --label-add zone=zone-1 <NODE1>
docker node update --label-add zone=zone-2 <NODE2>
docker node update --label-add zone=zone-3 <NODE3>

# Label storage nodes
docker node update --label-add storage-zone=zone-1 <NODE1>
docker node update --label-add storage-zone=zone-2 <NODE2>
docker node update --label-add storage-zone=zone-3 <NODE3>
```

### 2. Create Required Directories

```bash
# On each node, create storage directories
sudo mkdir -p /opt/hrp/storage/{gluster,minio}/{node1,node2,node3,node4}
sudo mkdir -p /opt/hrp/{postgres,redis,backups}
sudo chown -R 1000:1000 /opt/hrp/
```

### 3. Configure Environment Variables

```bash
# Copy and customize environment file
cp .env.example .env.production

# Edit the production environment file
nano .env.production
```

Required environment variables:
```env
# Database
POSTGRES_DB=hrp_database
POSTGRES_USER=hrp_user
POSTGRES_PASSWORD=your_secure_db_password
POSTGRES_REPLICATION_USER=replicator
POSTGRES_REPLICATION_PASSWORD=your_replication_password

# Redis
REDIS_PASSWORD=your_redis_password

# MinIO
MINIO_ROOT_USER=hrp_admin
MINIO_ROOT_PASSWORD=your_secure_minio_password

# Monitoring
GRAFANA_ADMIN_PASSWORD=your_grafana_password

# Alerts
WEBHOOK_URL=http://monitoring:9093/api/v1/alerts
ALERT_EMAIL=admin@yourdomain.com

# SSL (for production)
SSL_CERT_PATH=/etc/ssl/certs/hrp.crt
SSL_KEY_PATH=/etc/ssl/private/hrp.key
```

## Deployment Steps

### 1. Deploy Storage Layer

```bash
# Deploy distributed storage
docker stack deploy -c docker-compose.storage.yml hrp-storage

# Wait for storage to be ready
./scripts/storage/storage-monitor.sh monitor

# Verify storage health
./scripts/storage/storage-sync.sh check
```

### 2. Deploy Core Infrastructure

```bash
# Deploy the main application stack
docker stack deploy -c docker-swarm-stack.yml hrp

# Monitor deployment
docker stack ps hrp
docker service ls
```

### 3. Initialize Database

```bash
# Wait for PostgreSQL to be ready
sleep 60

# Initialize database schema
docker exec hrp_postgres-primary.1.$(docker service ps hrp_postgres-primary --format "{{.ID}}" | head -1) \
  psql -U hrp_user -d hrp_database -f /docker-entrypoint-initdb.d/01-setup-replication.sql

# Verify replication
docker exec hrp_postgres-replica.1.$(docker service ps hrp_postgres-replica --format "{{.ID}}" | head -1) \
  psql -U hrp_user -d hrp_database -c "SELECT * FROM pg_stat_replication;"
```

### 4. Configure Redis Cluster

```bash
# Verify Redis Sentinel
docker exec hrp_redis-sentinel.1.$(docker service ps hrp_redis-sentinel --format "{{.ID}}" | head -1) \
  redis-cli -p 26379 SENTINEL masters

# Test Redis failover
./scripts/failover/redis-failover.sh check
```

### 5. Verify Application Health

```bash
# Check all services
./scripts/failover/master-failover.sh status

# Test load balancer
curl -k https://your-domain.com/health

# Check monitoring
curl http://your-domain.com:3000  # Grafana
curl http://your-domain.com:9090  # Prometheus
```

## Post-Deployment Configuration

### 1. SSL Certificate Setup

```bash
# Copy SSL certificates to all nodes
scp your-cert.crt user@node:/opt/hrp/ssl/
scp your-cert.key user@node:/opt/hrp/ssl/

# Update HAProxy configuration
# Edit config/haproxy/haproxy.cfg to use your certificates
```

### 2. Monitoring Setup

```bash
# Access Grafana
# URL: https://your-domain.com:3000
# Username: admin
# Password: (from GRAFANA_ADMIN_PASSWORD)

# Import dashboards from config/grafana/dashboards/
# Configure alert channels in Grafana
```

### 3. Backup Configuration

```bash
# Test backup system
docker exec hrp_backup-service.1.$(docker service ps hrp_backup-service --format "{{.ID}}" | head -1) \
  /scripts/backup-postgres.sh

# Configure S3 backup (optional)
# Edit backup scripts to include S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
```

## Operational Procedures

### Health Monitoring

```bash
# Comprehensive health check
./scripts/failover/master-failover.sh check

# Individual service checks
./scripts/failover/postgres-failover.sh
./scripts/failover/redis-failover.sh check
./scripts/failover/app-failover.sh check
./scripts/storage/storage-monitor.sh monitor
```

### Manual Failover

```bash
# PostgreSQL failover
./scripts/failover/postgres-failover.sh

# Redis failover
./scripts/failover/redis-failover.sh failover

# Application failover
./scripts/failover/app-failover.sh failover

# Emergency failover (all services)
./scripts/failover/master-failover.sh emergency
```

### Scaling Operations

```bash
# Scale application instances
docker service scale hrp_app=8

# Scale specific services
docker service scale hrp_postgres-replica=2
docker service scale hrp_redis-replica=2
```

### Backup and Restore

```bash
# Manual backup
docker exec hrp_backup-service.1.$(docker service ps hrp_backup-service --format "{{.ID}}" | head -1) \
  /scripts/backup-postgres.sh

# Restore from backup
# 1. Stop application services
docker service scale hrp_app=0

# 2. Restore database
docker exec hrp_postgres-primary.1.$(docker service ps hrp_postgres-primary --format "{{.ID}}" | head -1) \
  pg_restore -U hrp_user -d hrp_database /backups/postgres/latest.sql

# 3. Restart services
docker service scale hrp_app=6
```

## Troubleshooting

### Common Issues

1. **Service Won't Start**
   ```bash
   # Check service logs
   docker service logs hrp_<service-name>
   
   # Check node resources
   docker node ls
   docker system df
   ```

2. **Database Connection Issues**
   ```bash
   # Check PostgreSQL status
   docker exec hrp_postgres-primary.1.$(docker service ps hrp_postgres-primary --format "{{.ID}}" | head -1) \
     pg_isready -U hrp_user
   
   # Check replication lag
   ./scripts/failover/postgres-failover.sh
   ```

3. **Storage Issues**
   ```bash
   # Check storage health
   ./scripts/storage/storage-monitor.sh monitor
   
   # Check GlusterFS status
   docker exec hrp-gluster-node-1 gluster volume status
   
   # Check MinIO status
   docker exec hrp-minio-1 mc admin info local
   ```

4. **High CPU/Memory Usage**
   ```bash
   # Check resource usage
   docker stats
   
   # Scale services if needed
   docker service scale hrp_app=8
   ```

### Log Locations

- Application logs: `/var/log/hrp/app.log`
- PostgreSQL logs: `/var/log/postgresql/`
- Redis logs: `/var/log/redis/`
- HAProxy logs: `/var/log/haproxy/`
- Backup logs: `/var/log/backup/`
- Failover logs: `/var/log/*-failover.log`
- Storage logs: `/var/log/storage-*.log`

## Security Considerations

1. **Network Security**
   - Use Docker overlay networks with encryption
   - Implement firewall rules between zones
   - Use VPN for inter-node communication

2. **Data Security**
   - Enable encryption at rest for databases
   - Use SSL/TLS for all communications
   - Rotate passwords regularly

3. **Access Control**
   - Implement RBAC for Docker Swarm
   - Use secrets management for sensitive data
   - Regular security audits

## Performance Tuning

### Database Optimization

```sql
-- PostgreSQL tuning (adjust based on your hardware)
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
SELECT pg_reload_conf();
```

### Redis Optimization

```bash
# Redis tuning in redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### Application Optimization

```bash
# Adjust worker processes based on CPU cores
# Edit app.py or use environment variables
export WORKERS=4
export WORKER_CONNECTIONS=1000
```

## Maintenance Procedures

### Regular Maintenance

1. **Weekly Tasks**
   - Review monitoring dashboards
   - Check backup integrity
   - Update security patches
   - Review log files

2. **Monthly Tasks**
   - Performance review and optimization
   - Capacity planning review
   - Disaster recovery testing
   - Security audit

3. **Quarterly Tasks**
   - Full disaster recovery drill
   - Infrastructure review
   - Update documentation
   - Review and update procedures

### Updates and Upgrades

```bash
# Application updates
docker service update --image hrp-app:new-version hrp_app

# Database updates (requires careful planning)
# 1. Test in staging environment
# 2. Schedule maintenance window
# 3. Backup before upgrade
# 4. Perform rolling upgrade

# Infrastructure updates
# Update Docker Swarm nodes one by one
# Ensure high availability during updates
```

## Support and Monitoring

### Monitoring URLs

- Grafana: `https://your-domain.com:3000`
- Prometheus: `https://your-domain.com:9090`
- HAProxy Stats: `https://your-domain.com:8404/stats`

### Alert Channels

Configure alerts for:
- Service failures
- High resource usage
- Database replication lag
- Storage issues
- Backup failures

### Contact Information

- System Administrator: admin@yourdomain.com
- On-call Engineer: oncall@yourdomain.com
- Emergency Contact: emergency@yourdomain.com

---

This deployment guide provides a comprehensive foundation for running the Horse Racing Prediction application with enterprise-grade high availability, monitoring, and disaster recovery capabilities.