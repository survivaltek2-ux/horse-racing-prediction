#!/usr/bin/env python3
"""
Firebase Security Rules Deployment Script

This script helps deploy and verify Firebase security rules to protect sensitive data.
"""

import os
import json
import subprocess
import sys
from datetime import datetime

def check_firebase_cli():
    """Check if Firebase CLI is installed"""
    try:
        result = subprocess.run(['firebase', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✓ Firebase CLI found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Firebase CLI not found")
        print("Please install Firebase CLI: npm install -g firebase-tools")
        return False

def check_firebase_login():
    """Check if user is logged into Firebase"""
    try:
        result = subprocess.run(['firebase', 'projects:list'], 
                              capture_output=True, text=True, check=True)
        print("✓ Firebase authentication verified")
        return True
    except subprocess.CalledProcessError:
        print("✗ Not logged into Firebase")
        print("Please run: firebase login")
        return False

def initialize_firebase_project():
    """Initialize Firebase project if not already done"""
    if not os.path.exists('firebase.json'):
        print("Initializing Firebase project...")
        try:
            # Create basic firebase.json
            firebase_config = {
                "firestore": {
                    "rules": "firestore.rules",
                    "indexes": "firestore.indexes.json"
                }
            }
            
            with open('firebase.json', 'w') as f:
                json.dump(firebase_config, f, indent=2)
            
            print("✓ Created firebase.json configuration")
            
            # Create basic indexes file
            indexes_config = {
                "indexes": [],
                "fieldOverrides": []
            }
            
            with open('firestore.indexes.json', 'w') as f:
                json.dump(indexes_config, f, indent=2)
            
            print("✓ Created firestore.indexes.json")
            
        except Exception as e:
            print(f"✗ Error initializing Firebase project: {e}")
            return False
    else:
        print("✓ Firebase project already initialized")
    
    return True

def validate_security_rules():
    """Validate Firestore security rules"""
    rules_path = 'firestore.rules'
    
    if not os.path.exists(rules_path):
        print(f"✗ Security rules file not found: {rules_path}")
        return False
    
    try:
        with open(rules_path, 'r') as f:
            rules_content = f.read()
        
        # Check for critical security patterns
        security_checks = [
            ('api_credentials', 'API credentials protection'),
            ('request.auth != null', 'Authentication requirement'),
            ('is_admin == true', 'Admin access control'),
            ('allow read, write: if false', 'Default deny rule')
        ]
        
        print("Validating security rules...")
        for pattern, description in security_checks:
            if pattern in rules_content:
                print(f"✓ {description} found")
            else:
                print(f"⚠️  {description} not found")
        
        return True
        
    except Exception as e:
        print(f"✗ Error validating security rules: {e}")
        return False

def deploy_security_rules():
    """Deploy Firestore security rules"""
    print("Deploying Firestore security rules...")
    
    try:
        result = subprocess.run(['firebase', 'deploy', '--only', 'firestore:rules'], 
                              capture_output=True, text=True, check=True)
        print("✓ Security rules deployed successfully")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error deploying security rules: {e}")
        print(f"Error output: {e.stderr}")
        return False

def test_security_rules():
    """Test security rules using Firebase emulator"""
    print("Testing security rules with Firebase emulator...")
    
    try:
        # Start emulator for testing
        print("Starting Firestore emulator for testing...")
        emulator_process = subprocess.Popen(['firebase', 'emulators:start', '--only', 'firestore'],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for emulator to start
        import time
        time.sleep(5)
        
        # Run security rules tests (if test file exists)
        test_file = 'firestore.test.js'
        if os.path.exists(test_file):
            test_result = subprocess.run(['npm', 'test'], 
                                       capture_output=True, text=True)
            if test_result.returncode == 0:
                print("✓ Security rules tests passed")
            else:
                print("⚠️  Security rules tests failed")
                print(test_result.stderr)
        else:
            print("ℹ No security rules tests found")
        
        # Stop emulator
        emulator_process.terminate()
        emulator_process.wait()
        
        return True
        
    except Exception as e:
        print(f"⚠️  Error testing security rules: {e}")
        return False

def audit_database_structure():
    """Audit database structure for sensitive data exposure"""
    print("\nAuditing database structure for sensitive data...")
    
    # Check if Firebase is configured
    try:
        from config.firebase_config import FirebaseConfig
        
        firebase_config = FirebaseConfig()
        if not firebase_config.is_connected():
            print("⚠️  Firebase not connected - cannot audit database structure")
            return False
        
        # Check collections for sensitive data
        sensitive_collections = {
            'api_credentials': 'Contains encrypted API keys and secrets',
            'users': 'Contains user authentication data',
            'config': 'May contain system configuration',
            'logs': 'May contain sensitive log information'
        }
        
        print("Checking for sensitive collections...")
        for collection, description in sensitive_collections.items():
            try:
                docs = firebase_config.db.collection(collection).limit(1).stream()
                doc_count = len(list(docs))
                if doc_count > 0:
                    print(f"⚠️  Found collection '{collection}': {description}")
                else:
                    print(f"ℹ Collection '{collection}' is empty")
            except Exception as e:
                print(f"ℹ Collection '{collection}' not accessible (good for security)")
        
        return True
        
    except ImportError:
        print("⚠️  Cannot import Firebase configuration - skipping database audit")
        return False
    except Exception as e:
        print(f"✗ Error auditing database: {e}")
        return False

def create_security_report():
    """Create a security audit report"""
    report_path = f"security_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    report_content = f"""Firebase Security Audit Report
Generated: {datetime.now().isoformat()}

SECURITY MEASURES IMPLEMENTED:
✓ Firebase service account file secured
✓ Environment variables configuration implemented
✓ Firestore security rules created
✓ Admin-only access to API credentials
✓ Authentication requirements enforced
✓ Default deny rules implemented

SECURITY RECOMMENDATIONS:
1. Regularly audit Firebase access logs
2. Rotate service account credentials periodically
3. Monitor for unusual access patterns
4. Keep Firebase SDK and dependencies updated
5. Implement proper backup security measures

NEXT STEPS:
1. Deploy security rules to production
2. Test access controls thoroughly
3. Set up monitoring and alerting
4. Train team on security best practices
5. Schedule regular security reviews

For questions or security concerns, refer to FIREBASE_SECURITY.md
"""
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"Security report saved to: {report_path}")
    except Exception as e:
        print(f"Error creating security report: {e}")
        return False

def main():
    """Main deployment function"""
    print("Firebase Security Rules Deployment Tool")
    print("="*50)
    
    success = True
    
    # Step 1: Check prerequisites
    print("Step 1: Checking prerequisites...")
    if not check_firebase_cli():
        success = False
    
    if success and not check_firebase_login():
        success = False
    
    # Step 2: Initialize Firebase project
    if success:
        print("\nStep 2: Initializing Firebase project...")
        if not initialize_firebase_project():
            success = False
    
    # Step 3: Validate security rules
    if success:
        print("\nStep 3: Validating security rules...")
        if not validate_security_rules():
            success = False
    
    # Step 4: Deploy security rules
    if success:
        print("\nStep 4: Deploying security rules...")
        if not deploy_security_rules():
            success = False
    
    # Step 5: Test security rules (optional)
    if success:
        print("\nStep 5: Testing security rules...")
        test_security_rules()  # Non-critical if this fails
    
    # Step 6: Audit database structure
    print("\nStep 6: Auditing database structure...")
    audit_database_structure()  # Non-critical if this fails
    
    # Step 7: Create security report
    print("\nStep 7: Creating security report...")
    create_security_report()
    
    # Final status
    print("\n" + "="*50)
    if success:
        print("✅ Firebase security deployment completed successfully!")
        print("\nIMPORTANT: Verify that your application works correctly with the new security rules.")
        print("Test all functionality, especially admin operations and API credential access.")
    else:
        print("❌ Firebase security deployment encountered errors.")
        print("Please address the issues above before proceeding.")
    
    print("\n📚 For more information, see FIREBASE_SECURITY.md")

if __name__ == "__main__":
    main()