#!/usr/bin/env python3
"""
Firebase Security Migration Script

This script helps migrate from insecure Firebase configuration (service account files)
to secure configuration (environment variables).

IMPORTANT: Run this script to secure your Firebase setup before deployment.
"""

import os
import json
import shutil
from datetime import datetime

def backup_service_account_file():
    """Create a secure backup of the service account file"""
    service_account_path = "config/firebase-service-account.json"
    
    if os.path.exists(service_account_path):
        # Create backup directory
        backup_dir = "backups/firebase"
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{backup_dir}/firebase-service-account-{timestamp}.json"
        
        shutil.copy2(service_account_path, backup_path)
        print(f"✓ Service account file backed up to: {backup_path}")
        
        # Set restrictive permissions on backup
        os.chmod(backup_path, 0o600)
        print("✓ Backup file permissions set to 600 (owner read/write only)")
        
        return backup_path
    else:
        print("ℹ No service account file found to backup")
        return None

def extract_credentials_from_file():
    """Extract credentials from service account file for environment variables"""
    service_account_path = "config/firebase-service-account.json"
    
    if not os.path.exists(service_account_path):
        print("ℹ No service account file found")
        return None
    
    try:
        with open(service_account_path, 'r') as f:
            credentials = json.load(f)
        
        extracted = {
            'FIREBASE_PROJECT_ID': credentials.get('project_id'),
            'FIREBASE_PRIVATE_KEY': credentials.get('private_key'),
            'FIREBASE_CLIENT_EMAIL': credentials.get('client_email')
        }
        
        print("✓ Credentials extracted from service account file")
        return extracted
    
    except Exception as e:
        print(f"✗ Error reading service account file: {e}")
        return None

def create_secure_env_template(credentials):
    """Create a secure .env template with extracted credentials"""
    if not credentials:
        print("ℹ No credentials to create template")
        return
    
    env_template = f"""# Firebase Configuration (SECURE)
# Use these environment variables instead of service account files

FIREBASE_PROJECT_ID={credentials['FIREBASE_PROJECT_ID']}
FIREBASE_PRIVATE_KEY="{credentials['FIREBASE_PRIVATE_KEY']}"
FIREBASE_CLIENT_EMAIL={credentials['FIREBASE_CLIENT_EMAIL']}

# Comment out or remove the service account path
# FIREBASE_SERVICE_ACCOUNT_PATH=config/firebase-service-account.json
"""
    
    # Write to a secure template file
    template_path = ".env.firebase.secure"
    with open(template_path, 'w') as f:
        f.write(env_template)
    
    # Set restrictive permissions
    os.chmod(template_path, 0o600)
    
    print(f"✓ Secure environment template created: {template_path}")
    print("ℹ Copy these values to your .env file and remove the service account file")

def update_gitignore():
    """Ensure .gitignore properly excludes sensitive files"""
    gitignore_additions = [
        "# Firebase Security",
        "firebase-service-account*.json",
        "config/firebase-service-account*.json",
        ".env.firebase.secure",
        "backups/firebase/",
        ""
    ]
    
    gitignore_path = ".gitignore"
    
    # Read existing .gitignore
    existing_content = ""
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            existing_content = f.read()
    
    # Check if Firebase security rules are already present
    if "firebase-service-account" not in existing_content:
        with open(gitignore_path, 'a') as f:
            f.write("\n" + "\n".join(gitignore_additions))
        print("✓ Updated .gitignore with Firebase security exclusions")
    else:
        print("ℹ .gitignore already contains Firebase security exclusions")

def verify_security_setup():
    """Verify that the security setup is correct"""
    print("\n" + "="*50)
    print("SECURITY VERIFICATION")
    print("="*50)
    
    issues = []
    
    # Check if service account file exists
    if os.path.exists("config/firebase-service-account.json"):
        issues.append("⚠️  Service account file still exists in config/")
    
    # Check if .env has Firebase environment variables
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            env_content = f.read()
        
        if "FIREBASE_PROJECT_ID" not in env_content:
            issues.append("⚠️  FIREBASE_PROJECT_ID not found in .env")
        if "FIREBASE_PRIVATE_KEY" not in env_content:
            issues.append("⚠️  FIREBASE_PRIVATE_KEY not found in .env")
        if "FIREBASE_CLIENT_EMAIL" not in env_content:
            issues.append("⚠️  FIREBASE_CLIENT_EMAIL not found in .env")
    else:
        issues.append("⚠️  .env file not found")
    
    # Check .gitignore
    if os.path.exists(".gitignore"):
        with open(".gitignore", 'r') as f:
            gitignore_content = f.read()
        
        if "firebase-service-account" not in gitignore_content:
            issues.append("⚠️  .gitignore doesn't exclude Firebase service account files")
    else:
        issues.append("⚠️  .gitignore file not found")
    
    if issues:
        print("SECURITY ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print("\nPlease address these issues before deployment!")
        return False
    else:
        print("✅ Security verification passed!")
        print("✅ Firebase configuration is secure for deployment")
        return True

def main():
    """Main migration function"""
    print("Firebase Security Migration Tool")
    print("="*50)
    print("This tool helps secure your Firebase configuration")
    print("by migrating from service account files to environment variables.\n")
    
    # Step 1: Backup existing service account file
    print("Step 1: Backing up service account file...")
    backup_path = backup_service_account_file()
    
    # Step 2: Extract credentials
    print("\nStep 2: Extracting credentials...")
    credentials = extract_credentials_from_file()
    
    # Step 3: Create secure environment template
    print("\nStep 3: Creating secure environment template...")
    create_secure_env_template(credentials)
    
    # Step 4: Update .gitignore
    print("\nStep 4: Updating .gitignore...")
    update_gitignore()
    
    # Step 5: Verify security setup
    print("\nStep 5: Verifying security setup...")
    is_secure = verify_security_setup()
    
    # Final instructions
    print("\n" + "="*50)
    print("NEXT STEPS")
    print("="*50)
    
    if credentials:
        print("1. Copy the values from .env.firebase.secure to your .env file")
        print("2. Test your application with the new environment variables")
        print("3. Remove or secure the service account file:")
        print("   - For local development: move to a secure location outside the project")
        print("   - For production: delete the file completely")
        print("4. Deploy Firestore security rules using: firebase deploy --only firestore:rules")
        print("5. Verify that Firebase authentication is properly configured")
    
    print("\n⚠️  IMPORTANT SECURITY REMINDERS:")
    print("   - Never commit service account files to version control")
    print("   - Use environment variables in production")
    print("   - Implement proper Firestore security rules")
    print("   - Regularly audit Firebase access and permissions")
    
    if backup_path:
        print(f"\n📁 Service account backup saved to: {backup_path}")
        print("   Keep this backup in a secure location for recovery purposes")

if __name__ == "__main__":
    main()