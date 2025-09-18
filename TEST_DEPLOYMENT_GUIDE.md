# Test Deployment Guide
## Horse Racing Prediction Application

This guide provides step-by-step instructions for performing a safe test deployment of the Horse Racing Prediction application.

---

## 📋 1. Pre-Deployment Checks

### 1.1 System Requirements Verification
```bash
# Check Docker installation
docker --version
docker-compose --version

# Check available disk space (minimum 5GB recommended)
df -h

# Check available memory (minimum 2GB recommended)
free -h

# Verify network connectivity
ping -c 3 google.com
```

### 1.2 Application Health Check
```bash
# Navigate to application directory
cd /Users/richardsiebert/HorseRacingPrediction/APP

# Verify all required files exist
ls -la .env.production docker-compose.production.yml deploy.sh

# Check for any uncommitted changes
git status

# Verify Python dependencies
python3 -m pip check
```

### 1.3 Security Pre-Checks
```bash
# Ensure no sensitive data in logs
find logs/ -name "*.log" -exec grep -l "password\|secret\|key" {} \; 2>/dev/null

# Check file permissions
ls -la .env* *.sh

# Verify no debug mode in configuration
grep -r "DEBUG.*True" config/ || echo "✅ No debug mode found"
```

### 1.4 Backup Current State
```bash
# Create backup directory with timestamp
BACKUP_DIR="backups/pre-deployment-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup database
cp data/*.json "$BACKUP_DIR/" 2>/dev/null || echo "No JSON data files to backup"

# Backup configuration
cp .env* "$BACKUP_DIR/"

# Backup logs
cp -r logs/ "$BACKUP_DIR/" 2>/dev/null || echo "No logs to backup"

echo "✅ Backup created in $BACKUP_DIR"
```

---

## 🔧 2. Environment Setup Instructions

### 2.1 Environment Configuration
```bash
# Copy production environment template
cp .env.production .env.test

# Edit test environment variables
nano .env.test
```

**Required Environment Variables for Test:**
```bash
# Flask Configuration
FLASK_ENV=production
SECRET_KEY=test-secret-key-change-for-production
DEBUG=False

# Database Configuration (Use test database)
DATABASE_URL=sqlite:///test_database.db
DATABASE_ECHO=False

# Test API Configuration
API_TIMEOUT=30
API_MAX_RETRIES=3
API_RATE_LIMIT_PER_MINUTE=60
DEFAULT_API_PROVIDER=mock

# Security Configuration
SESSION_COOKIE_SECURE=False  # Set to True for HTTPS
SESSION_COOKIE_HTTPONLY=True
SESSION_COOKIE_SAMESITE=Lax
PERMANENT_SESSION_LIFETIME=3600

# Server Configuration
HOST=127.0.0.1
PORT=8001
WORKERS=2

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/test_deployment.log
```

### 2.2 Test Database Setup
```bash
# Create test database directory
mkdir -p test_data

# Initialize test database
python3 -c "
from app import app, db
with app.app_context():
    db.create_all()
    print('✅ Test database initialized')
"

# Create test admin user
python3 create_admin_user.py
```

### 2.3 SSL Certificate Setup (Optional for Test)
```bash
# Create self-signed certificates for testing
mkdir -p ssl
openssl req -x509 -newkey rsa:4096 -keyout ssl/test-key.pem -out ssl/test-cert.pem -days 30 -nodes -subj "/C=US/ST=Test/L=Test/O=Test/CN=localhost"

echo "✅ Test SSL certificates created"
```

---

## 🚀 3. Deployment Commands/Process

### 3.1 Docker Test Deployment
```bash
# Create test docker-compose file
cp docker-compose.production.yml docker-compose.test.yml

# Modify for test environment
sed -i 's/.env.production/.env.test/g' docker-compose.test.yml
sed -i 's/hrp-app-prod/hrp-app-test/g' docker-compose.test.yml
sed -i 's/8000:8000/8001:8001/g' docker-compose.test.yml

# Build and start test containers
echo "🚀 Starting test deployment..."
docker-compose -f docker-compose.test.yml build --no-cache

# Start services in detached mode
docker-compose -f docker-compose.test.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30
```

### 3.2 Manual Test Deployment (Alternative)
```bash
# Create virtual environment for test
python3 -m venv test_venv
source test_venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export FLASK_ENV=production
export DEBUG=False
export PORT=8001

# Start application
echo "🚀 Starting test application..."
gunicorn --bind 127.0.0.1:8001 --workers 2 app:app &
TEST_PID=$!

echo "✅ Test application started with PID: $TEST_PID"
echo "$TEST_PID" > test_deployment.pid
```

### 3.3 Service Health Monitoring
```bash
# Monitor container status
watch -n 5 'docker-compose -f docker-compose.test.yml ps'

# Monitor application logs
docker-compose -f docker-compose.test.yml logs -f app &
LOG_PID=$!
```

---

## ✅ 4. Post-Deployment Verification Steps

### 4.1 Service Availability Check
```bash
# Wait for application to be ready
echo "🔍 Checking service availability..."
for i in {1..30}; do
    if curl -f http://127.0.0.1:8001/ >/dev/null 2>&1; then
        echo "✅ Application is responding"
        break
    fi
    echo "⏳ Waiting for application... ($i/30)"
    sleep 2
done
```

### 4.2 Endpoint Testing
```bash
# Test main endpoints
echo "🧪 Testing application endpoints..."

# Test home page
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8001/ | grep -q "200" && echo "✅ Home page: OK" || echo "❌ Home page: FAILED"

# Test API endpoints
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8001/api/providers | grep -q "200" && echo "✅ API providers: OK" || echo "❌ API providers: FAILED"

# Test static files
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8001/static/css/style.css | grep -q "200" && echo "✅ Static files: OK" || echo "❌ Static files: FAILED"

# Test database connectivity
curl -s http://127.0.0.1:8001/races | grep -q "races" && echo "✅ Database: OK" || echo "❌ Database: FAILED"
```

### 4.3 Security Verification
```bash
echo "🔒 Performing security checks..."

# Check for debug mode
curl -s http://127.0.0.1:8001/ | grep -q "debug" && echo "❌ Debug mode detected" || echo "✅ Debug mode: OFF"

# Check security headers
curl -I http://127.0.0.1:8001/ | grep -q "X-Content-Type-Options" && echo "✅ Security headers: OK" || echo "⚠️ Security headers: Missing"

# Check for sensitive information exposure
curl -s http://127.0.0.1:8001/ | grep -E "(password|secret|key)" && echo "❌ Sensitive data exposed" || echo "✅ No sensitive data exposed"
```

### 4.4 Performance Testing
```bash
echo "⚡ Running performance tests..."

# Test response time
RESPONSE_TIME=$(curl -o /dev/null -s -w "%{time_total}" http://127.0.0.1:8001/)
echo "📊 Response time: ${RESPONSE_TIME}s"

# Test concurrent requests
echo "🔄 Testing concurrent requests..."
for i in {1..10}; do
    curl -s http://127.0.0.1:8001/ >/dev/null &
done
wait
echo "✅ Concurrent requests test completed"

# Check memory usage
if command -v docker >/dev/null; then
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep hrp-app-test
fi
```

### 4.5 Log Analysis
```bash
echo "📋 Analyzing application logs..."

# Check for errors in logs
if [ -f logs/test_deployment.log ]; then
    ERROR_COUNT=$(grep -c "ERROR" logs/test_deployment.log)
    WARNING_COUNT=$(grep -c "WARNING" logs/test_deployment.log)
    echo "📊 Errors: $ERROR_COUNT, Warnings: $WARNING_COUNT"
    
    if [ $ERROR_COUNT -gt 0 ]; then
        echo "❌ Errors found in logs:"
        grep "ERROR" logs/test_deployment.log | tail -5
    fi
fi

# Check Docker logs if using containers
if command -v docker >/dev/null; then
    echo "🐳 Recent Docker logs:"
    docker-compose -f docker-compose.test.yml logs --tail=20 app
fi
```

---

## 🔄 5. Rollback Procedure

### 5.1 Immediate Rollback (Emergency)
```bash
echo "🚨 Initiating emergency rollback..."

# Stop test deployment
if [ -f docker-compose.test.yml ]; then
    docker-compose -f docker-compose.test.yml down
    echo "✅ Docker containers stopped"
fi

# Stop manual deployment
if [ -f test_deployment.pid ]; then
    PID=$(cat test_deployment.pid)
    kill $PID 2>/dev/null && echo "✅ Application process stopped"
    rm test_deployment.pid
fi

# Kill log monitoring
if [ ! -z "$LOG_PID" ]; then
    kill $LOG_PID 2>/dev/null
fi
```

### 5.2 Data Restoration
```bash
echo "💾 Restoring data from backup..."

# Find latest backup
LATEST_BACKUP=$(ls -t backups/pre-deployment-* | head -1)
echo "📁 Using backup: $LATEST_BACKUP"

# Restore database
if [ -d "$LATEST_BACKUP" ]; then
    cp "$LATEST_BACKUP"/*.json data/ 2>/dev/null || echo "No data files to restore"
    cp "$LATEST_BACKUP"/.env* . 2>/dev/null || echo "No env files to restore"
    echo "✅ Data restored from backup"
fi
```

### 5.3 Environment Cleanup
```bash
echo "🧹 Cleaning up test environment..."

# Remove test files
rm -f .env.test docker-compose.test.yml
rm -f test_database.db
rm -rf test_data/
rm -rf test_venv/
rm -rf ssl/test-*

# Clean up Docker resources
docker system prune -f
docker volume prune -f

echo "✅ Test environment cleaned up"
```

### 5.4 Verification of Rollback
```bash
echo "🔍 Verifying rollback completion..."

# Check no test processes running
ps aux | grep -v grep | grep -q "test\|8001" && echo "⚠️ Test processes still running" || echo "✅ No test processes found"

# Check port availability
lsof -i :8001 && echo "⚠️ Port 8001 still in use" || echo "✅ Port 8001 available"

# Verify original environment
[ -f .env.production ] && echo "✅ Production environment preserved" || echo "❌ Production environment missing"

echo "✅ Rollback completed successfully"
```

---

## 🛡️ Security Measures

### During Deployment:
- ✅ Use non-privileged user accounts
- ✅ Disable debug mode in production
- ✅ Use secure environment variables
- ✅ Implement proper file permissions
- ✅ Monitor for sensitive data exposure

### Network Security:
- ✅ Use HTTPS in production (HTTP acceptable for local testing)
- ✅ Implement proper CORS policies
- ✅ Use secure session configuration
- ✅ Monitor for unauthorized access attempts

### Data Protection:
- ✅ Create backups before deployment
- ✅ Use separate test database
- ✅ Encrypt sensitive configuration data
- ✅ Implement proper access controls

---

## 📞 Troubleshooting

### Common Issues:

**Port Already in Use:**
```bash
lsof -i :8001
kill -9 <PID>
```

**Database Connection Issues:**
```bash
# Check database file permissions
ls -la *.db
# Reinitialize if needed
rm test_database.db && python3 init_database.py
```

**Docker Issues:**
```bash
# Reset Docker environment
docker-compose -f docker-compose.test.yml down -v
docker system prune -f
```

**Memory Issues:**
```bash
# Check system resources
free -h
df -h
# Reduce worker count in configuration
```

---

## 📋 Test Deployment Checklist

- [ ] Pre-deployment checks completed
- [ ] Environment configured
- [ ] Backup created
- [ ] Application deployed
- [ ] Service availability verified
- [ ] Endpoints tested
- [ ] Security checks passed
- [ ] Performance acceptable
- [ ] Logs reviewed
- [ ] Rollback procedure tested

---

**⚠️ Important Notes:**
- Always test rollback procedures before production deployment
- Monitor system resources during deployment
- Keep detailed logs of all deployment activities
- Have emergency contacts ready
- Test in an environment as close to production as possible

**📞 Emergency Contacts:**
- System Administrator: [Contact Info]
- Database Administrator: [Contact Info]
- Security Team: [Contact Info]

---

*Last Updated: $(date)*
*Version: 1.0*