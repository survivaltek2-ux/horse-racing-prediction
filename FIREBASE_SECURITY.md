# Firebase Security Guidelines

## CRITICAL SECURITY ALERT

**⚠️ NEVER COMMIT SERVICE ACCOUNT FILES TO VERSION CONTROL ⚠️**

Firebase service account files contain private keys and sensitive credentials that should never be exposed publicly. This document outlines secure practices for Firebase configuration.

## Security Issues Identified

1. **Service Account File Exposure**: The `firebase-service-account.json` file contains sensitive credentials
2. **Public API Credentials**: API credentials stored in Firebase collections need proper security rules
3. **Insecure Configuration**: Service account files should not be used in production

## Secure Configuration Methods

### Method 1: Environment Variables (RECOMMENDED for Production)

Use environment variables to store Firebase credentials securely:

```bash
# .env file (never commit this file)
FIREBASE_PROJECT_ID=your-firebase-project-id
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----"
FIREBASE_CLIENT_EMAIL=firebase-adminsdk-xxxxx@your-project.iam.gserviceaccount.com
```

### Method 2: Service Account File (Local Development Only)

For local development only, you can use a service account file:

```bash
# .env file
FIREBASE_SERVICE_ACCOUNT_PATH=config/firebase-service-account.json
```

**Important**: 
- Never commit the service account file to version control
- Use `.gitignore` to exclude these files
- Only use this method for local development

### Method 3: Default Credentials (Google Cloud Environment)

When running on Google Cloud Platform, use default credentials:
- No additional configuration needed
- Automatically uses the service account attached to the compute instance

## Firebase Security Rules

Implement proper Firestore security rules to protect sensitive data:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Admin-only access to API credentials
    match /api_credentials/{document} {
      allow read, write: if request.auth != null && 
        get(/databases/$(database)/documents/users/$(request.auth.uid)).data.is_admin == true;
    }
    
    // User authentication required for sensitive operations
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
    
    // Public read access for non-sensitive data
    match /races/{document} {
      allow read: if true;
      allow write: if request.auth != null;
    }
    
    match /horses/{document} {
      allow read: if true;
      allow write: if request.auth != null;
    }
    
    // Predictions - users can read all, but only create/modify their own
    match /predictions/{document} {
      allow read: if true;
      allow create: if request.auth != null;
      allow update, delete: if request.auth != null && 
        resource.data.user_id == request.auth.uid;
    }
  }
}
```

## Deployment Security Checklist

### Before Deployment:

- [ ] Remove or secure all service account files
- [ ] Use environment variables for Firebase credentials
- [ ] Implement proper Firestore security rules
- [ ] Enable Firebase Authentication
- [ ] Review and restrict API access
- [ ] Set up proper IAM roles and permissions
- [ ] Enable audit logging
- [ ] Configure network security (if applicable)

### Production Environment:

- [ ] Use environment variables only
- [ ] Never use service account files
- [ ] Implement least privilege access
- [ ] Regular security audits
- [ ] Monitor access logs
- [ ] Rotate credentials regularly

## Migration Steps

### Step 1: Secure Current Setup

1. **Immediately**: Ensure service account files are in `.gitignore`
2. **Extract credentials**: Copy values from service account file to environment variables
3. **Test**: Verify application works with environment variables
4. **Remove**: Delete or move service account files to secure location

### Step 2: Update Configuration

1. Update `.env` file with Firebase environment variables
2. Test Firebase initialization with new configuration
3. Verify all Firebase operations work correctly

### Step 3: Implement Security Rules

1. Deploy Firestore security rules
2. Test access controls
3. Verify admin-only access to sensitive collections

## Environment Variable Setup

### For Production Deployment:

```bash
# Set these in your production environment
export FIREBASE_PROJECT_ID="your-firebase-project-id"
export FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----"
export FIREBASE_CLIENT_EMAIL="firebase-adminsdk-xxxxx@your-project.iam.gserviceaccount.com"
```

### For Docker Deployment:

```dockerfile
# In your Dockerfile or docker-compose.yml
ENV FIREBASE_PROJECT_ID=your-firebase-project-id
ENV FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----"
ENV FIREBASE_CLIENT_EMAIL=firebase-adminsdk-xxxxx@your-project.iam.gserviceaccount.com
```

## Security Best Practices

1. **Never commit credentials**: Use `.gitignore` and environment variables
2. **Principle of least privilege**: Grant minimum necessary permissions
3. **Regular audits**: Review access logs and permissions regularly
4. **Credential rotation**: Rotate service account keys periodically
5. **Monitoring**: Set up alerts for unusual access patterns
6. **Backup security**: Ensure backups don't contain exposed credentials

## Incident Response

If credentials are accidentally exposed:

1. **Immediate**: Revoke the exposed service account key
2. **Generate**: Create new service account key
3. **Update**: Update all environments with new credentials
4. **Audit**: Review access logs for unauthorized usage
5. **Monitor**: Watch for suspicious activity

## Contact

For security issues or questions, please follow responsible disclosure practices and contact the development team immediately.

---

**Remember**: Security is everyone's responsibility. When in doubt, choose the more secure option.