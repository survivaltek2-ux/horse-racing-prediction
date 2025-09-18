# Secure SSH Deployment System

A comprehensive, secure SSH deployment solution for the Horse Racing Prediction application that transfers files to remote droplets with integrity verification and robust error handling.

## 🔐 Security Features

- **SSH Key Authentication**: Uses SSH keys instead of passwords for enhanced security
- **Encrypted File Transfer**: All data is encrypted during transfer using SSH/SCP/rsync
- **File Integrity Verification**: SHA256 checksums ensure files are transferred correctly
- **Secure Permissions**: Automatically sets appropriate file permissions on the remote server
- **Connection Security**: Implements timeouts, keep-alive, and secure connection options
- **Comprehensive Logging**: Detailed logs for audit trails and troubleshooting

## 📁 Files Overview

- `secure-ssh-deploy.sh` - Main deployment script with all security features
- `deploy-via-ssh.sh` - User-friendly wrapper script with configuration loading
- `ssh-deploy-config.env` - Configuration template file
- `SSH_DEPLOYMENT_README.md` - This documentation file

## 🚀 Quick Start

### 1. Setup Configuration

Copy the configuration template and customize it:

```bash
cd scripts/deployment/
cp ssh-deploy-config.env ssh-deploy-config.local.env
nano ssh-deploy-config.local.env
```

### 2. Configure Your Settings

Edit `ssh-deploy-config.local.env` with your server details:

```bash
# Remote server configuration
REMOTE_HOST="your-droplet-ip.com"
REMOTE_USER="root"
REMOTE_APP_DIR="/opt/horse-racing-app"

# SSH configuration
SSH_KEY_PATH="~/.ssh/id_rsa"
```

### 3. Run Deployment

Execute the deployment:

```bash
./deploy-via-ssh.sh
```

## 📋 Prerequisites

### SSH Key Setup

1. **Generate SSH Key** (if you don't have one):
   ```bash
   ssh-keygen -t rsa -b 4096 -C "your-email@example.com"
   ```

2. **Copy Public Key to Remote Server**:
   ```bash
   ssh-copy-id -i ~/.ssh/id_rsa.pub user@your-server.com
   ```

3. **Test SSH Connection**:
   ```bash
   ssh -i ~/.ssh/id_rsa user@your-server.com
   ```

### Remote Server Requirements

- SSH server running and accessible
- User account with appropriate permissions
- Sufficient disk space for the application
- Network connectivity from your local machine

## ⚙️ Configuration Options

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `REMOTE_HOST` | Server hostname or IP | `"192.168.1.100"` |
| `REMOTE_USER` | SSH username | `"root"` or `"ubuntu"` |
| `SSH_KEY_PATH` | Path to SSH private key | `"~/.ssh/id_rsa"` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REMOTE_APP_DIR` | Remote application directory | `/opt/horse-racing-app` |
| `SSH_PORT` | SSH port number | `22` |
| `SSH_TIMEOUT` | Connection timeout (seconds) | `30` |

## 🔄 Deployment Process

The deployment script performs the following steps:

1. **Configuration Validation**
   - Validates all required environment variables
   - Checks SSH key existence and permissions
   - Corrects SSH key permissions if needed (sets to 600)

2. **SSH Connectivity Test**
   - Tests SSH connection to remote server
   - Validates authentication and network connectivity
   - Implements connection timeouts and retry logic

3. **Remote Directory Setup**
   - Creates application directory structure
   - Sets appropriate ownership and permissions
   - Creates subdirectories for config, data, logs, backups

4. **File Transfer**
   - Uses rsync for efficient, resumable transfers
   - Shows progress during transfer
   - Excludes unnecessary files (.git, __pycache__, etc.)
   - Preserves file permissions and timestamps

5. **Deployment Configuration Transfer**
   - Transfers critical deployment files:
     - Dockerfiles
     - docker-compose files
     - Environment files (.env.production)
     - Configuration files
     - Deployment scripts
     - Nginx configurations

6. **Permission Setting**
   - Sets secure permissions on all files
   - Makes scripts executable
   - Secures sensitive files (600 permissions)
   - Sets directory permissions appropriately

7. **Integrity Verification**
   - Calculates SHA256 checksums for all files
   - Compares local and remote checksums
   - Reports any discrepancies
   - Ensures complete and accurate transfer

## 📊 File Transfer Details

### Included Files
- All application source code
- Configuration files
- Static assets
- Templates and views
- Requirements and dependencies
- Deployment scripts
- Documentation

### Excluded Files
- `.git/` directory
- `__pycache__/` directories
- `node_modules/` directories
- `venv/` virtual environments
- Local environment files (`.env.local`)
- Log files
- Temporary files

### Special Handling
- **Hidden Files**: Included (like `.env.production`)
- **Executable Scripts**: Permissions preserved
- **Sensitive Files**: Secured with 600 permissions
- **Large Files**: Transferred with progress indication

## 🛡️ Security Best Practices

### SSH Security
- Uses SSH key authentication (no passwords)
- Implements connection timeouts
- Disables strict host key checking for automation
- Uses secure SSH options and ciphers

### File Security
- Verifies file integrity with checksums
- Sets minimal required permissions
- Secures sensitive configuration files
- Excludes development and temporary files

### Network Security
- Encrypted data transfer
- Connection keep-alive for stability
- Timeout handling for unreliable networks
- Batch mode for non-interactive operation

## 🔧 Troubleshooting

### Common Issues

#### SSH Connection Failed
```bash
# Check SSH key permissions
ls -la ~/.ssh/id_rsa
chmod 600 ~/.ssh/id_rsa

# Test SSH connection manually
ssh -i ~/.ssh/id_rsa user@server.com

# Check if SSH service is running on remote server
telnet server.com 22
```

#### Permission Denied
```bash
# Ensure SSH key is added to remote server
ssh-copy-id -i ~/.ssh/id_rsa.pub user@server.com

# Check remote user permissions
ssh user@server.com "sudo -l"
```

#### File Transfer Interrupted
```bash
# The script uses rsync which supports resume
# Simply run the deployment again - it will resume from where it left off
./deploy-via-ssh.sh
```

#### Integrity Verification Failed
```bash
# Check network stability
ping server.com

# Check disk space on remote server
ssh user@server.com "df -h"

# Re-run deployment to retry transfer
./deploy-via-ssh.sh
```

### Log Analysis

Check the deployment log for detailed information:
```bash
# Log file location is shown during deployment
tail -f /tmp/secure-ssh-deploy-YYYYMMDD-HHMMSS.log
```

## 📈 Advanced Usage

### Custom SSH Options

You can modify the SSH options in the script for specific requirements:

```bash
# Example: Custom port and cipher
ssh_opts=(
    -i "$SSH_KEY_PATH"
    -p 2222
    -c aes256-ctr
    -o StrictHostKeyChecking=no
    -o ConnectTimeout=30
)
```

### Multiple Environment Deployment

Create different configuration files for different environments:

```bash
# Production
cp ssh-deploy-config.env ssh-deploy-config.production.env

# Staging
cp ssh-deploy-config.env ssh-deploy-config.staging.env

# Use specific config
source ssh-deploy-config.production.env
./secure-ssh-deploy.sh
```

### Automated Deployment

For CI/CD integration:

```bash
#!/bin/bash
# Set environment variables
export REMOTE_HOST="production.server.com"
export REMOTE_USER="deploy"
export SSH_KEY_PATH="/path/to/deploy/key"
export REMOTE_APP_DIR="/var/www/app"

# Run deployment
./scripts/deployment/secure-ssh-deploy.sh
```

## 🔍 Monitoring and Maintenance

### Post-Deployment Verification

After deployment, verify the application:

```bash
# SSH into the server
ssh -i ~/.ssh/id_rsa user@server.com

# Check application files
cd /opt/horse-racing-app
ls -la

# Verify permissions
find . -type f -name "*.sh" -exec ls -la {} \;

# Check deployment logs
tail -f logs/deployment.log
```

### Regular Maintenance

- Monitor log files for any issues
- Keep SSH keys secure and rotated
- Update the deployment script as needed
- Test deployment process regularly
- Backup configuration files

## 📞 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the deployment logs
3. Test SSH connectivity manually
4. Verify server requirements are met
5. Check network connectivity and firewall settings

## 🔄 Version History

- **v1.0.0**: Initial release with full security features
  - SSH key authentication
  - File integrity verification
  - Progress feedback
  - Comprehensive error handling
  - Secure permission setting
  - Detailed logging