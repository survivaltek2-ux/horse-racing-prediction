#!/usr/bin/env python3
"""
Firebase Security Audit Script

This script performs a comprehensive security audit of the Firebase configuration
and database structure to identify potential vulnerabilities.

Run this script regularly to ensure your Firebase setup remains secure.
"""

import os
import sys
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

def check_environment_variables() -> Dict[str, Any]:
    """Check Firebase environment variable configuration"""
    print("\n1. Environment Variables Security Check")
    print("-" * 50)
    
    results = {
        "status": "pass",
        "issues": [],
        "recommendations": []
    }
    
    # Check for secure environment variable setup
    firebase_project_id = os.getenv('FIREBASE_PROJECT_ID')
    firebase_private_key = os.getenv('FIREBASE_PRIVATE_KEY')
    firebase_client_email = os.getenv('FIREBASE_CLIENT_EMAIL')
    service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
    
    if firebase_project_id and firebase_private_key and firebase_client_email:
        print("✓ Environment variables configured (secure method)")
    elif service_account_path:
        print("⚠️  Using service account file (less secure)")
        results["issues"].append("Using service account file instead of environment variables")
        results["recommendations"].append("Migrate to environment variables for production")
        results["status"] = "warning"
    else:
        print("✗ No Firebase configuration found")
        results["issues"].append("No Firebase configuration detected")
        results["status"] = "fail"
    
    # Check for encryption key
    encryption_key = os.getenv('ENCRYPTION_KEY')
    if encryption_key:
        if len(encryption_key) >= 32:
            print("✓ Encryption key configured with adequate length")
        else:
            print("⚠️  Encryption key too short")
            results["issues"].append("Encryption key length insufficient")
            results["status"] = "warning"
    else:
        print("✗ No encryption key configured")
        results["issues"].append("No encryption key found")
        results["status"] = "fail"
    
    return results

def check_service_account_files() -> Dict[str, Any]:
    """Check for exposed service account files"""
    print("\n2. Service Account File Security Check")
    print("-" * 50)
    
    results = {
        "status": "pass",
        "issues": [],
        "recommendations": []
    }
    
    # Check for service account files in common locations
    service_account_patterns = [
        "config/firebase-service-account.json",
        "firebase-service-account.json",
        "*service-account*.json"
    ]
    
    found_files = []
    for pattern in service_account_patterns:
        if '*' in pattern:
            # Use glob for wildcard patterns
            import glob
            files = glob.glob(pattern, recursive=True)
            found_files.extend(files)
        else:
            if os.path.exists(pattern):
                found_files.append(pattern)
    
    if found_files:
        print("⚠️  Service account files found:")
        for file in found_files:
            print(f"    - {file}")
        results["issues"].append(f"Service account files found: {', '.join(found_files)}")
        results["recommendations"].append("Move service account files to secure location")
        results["recommendations"].append("Use environment variables instead")
        results["status"] = "warning"
    else:
        print("✓ No service account files found in project directory")
    
    return results

def check_gitignore_rules() -> Dict[str, Any]:
    """Check .gitignore for proper exclusion rules"""
    print("\n3. Version Control Security Check")
    print("-" * 50)
    
    results = {
        "status": "pass",
        "issues": [],
        "recommendations": []
    }
    
    gitignore_path = ".gitignore"
    if not os.path.exists(gitignore_path):
        print("✗ No .gitignore file found")
        results["issues"].append("No .gitignore file")
        results["status"] = "fail"
        return results
    
    with open(gitignore_path, 'r') as f:
        gitignore_content = f.read()
    
    # Check for essential security patterns
    security_patterns = [
        r'\*service-account\*\.json',
        r'firebase-service-account\.json',
        r'\.env',
        r'\*\.key',
        r'\*\.pem'
    ]
    
    missing_patterns = []
    for pattern in security_patterns:
        if not re.search(pattern, gitignore_content):
            missing_patterns.append(pattern)
    
    if missing_patterns:
        print("⚠️  Missing .gitignore patterns:")
        for pattern in missing_patterns:
            print(f"    - {pattern}")
        results["issues"].append(f"Missing security patterns: {', '.join(missing_patterns)}")
        results["status"] = "warning"
    else:
        print("✓ .gitignore contains essential security patterns")
    
    return results

def check_code_for_hardcoded_secrets() -> Dict[str, Any]:
    """Check code files for hardcoded secrets"""
    print("\n4. Hardcoded Secrets Check")
    print("-" * 50)
    
    results = {
        "status": "pass",
        "issues": [],
        "recommendations": []
    }
    
    # Patterns that might indicate hardcoded secrets
    secret_patterns = [
        r'api_key\s*=\s*["\'][^"\']+["\']',
        r'api_secret\s*=\s*["\'][^"\']+["\']',
        r'password\s*=\s*["\'][^"\']+["\']',
        r'private_key\s*=\s*["\'][^"\']+["\']',
        r'firebase.*["\'][A-Za-z0-9+/=]{20,}["\']'
    ]
    
    # Files to check
    file_extensions = ['.py', '.js', '.json', '.yaml', '.yml']
    suspicious_files = []
    
    # Files to exclude from secret detection (templates, migration scripts, etc.)
    exclude_files = [
        'secure_firebase_migration.py',
        'firebase_security_audit.py',
        'deploy_firebase_security.py',
        '.template',
        '.example'
    ]
    
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.env']]
        
        for file in files:
            # Skip excluded files
            if any(exclude in file for exclude in exclude_files):
                continue
                
            if any(file.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        for pattern in secret_patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                suspicious_files.append(file_path)
                                break
                except Exception:
                    continue
    
    if suspicious_files:
        print("⚠️  Files with potential hardcoded secrets:")
        for file in suspicious_files:
            print(f"    - {file}")
        results["issues"].append(f"Potential hardcoded secrets in: {', '.join(suspicious_files)}")
        results["recommendations"].append("Review files for hardcoded credentials")
        results["status"] = "warning"
    else:
        print("✓ No obvious hardcoded secrets found")
    
    return results

def check_firebase_security_rules() -> Dict[str, Any]:
    """Check Firebase security rules"""
    print("\n5. Firebase Security Rules Check")
    print("-" * 50)
    
    results = {
        "status": "pass",
        "issues": [],
        "recommendations": []
    }
    
    rules_file = "firestore.rules"
    if not os.path.exists(rules_file):
        print("✗ No firestore.rules file found")
        results["issues"].append("No Firestore security rules file")
        results["status"] = "fail"
        return results
    
    with open(rules_file, 'r') as f:
        rules_content = f.read()
    
    # Check for essential security patterns in rules
    security_checks = [
        (r'api_credentials.*\n.*is_admin\s*==\s*true', "API credentials protected by admin check"),
        (r'request\.auth\s*!=\s*null', "Authentication required"),
        (r'allow\s+read,\s*write:\s*if\s+false', "Default deny rule")
    ]
    
    passed_checks = 0
    for pattern, description in security_checks:
        if re.search(pattern, rules_content, re.DOTALL):
            print(f"✓ {description}")
            passed_checks += 1
        else:
            print(f"⚠️  Missing: {description}")
            results["issues"].append(f"Missing security rule: {description}")
    
    if passed_checks < len(security_checks):
        results["status"] = "warning"
        results["recommendations"].append("Review and enhance Firestore security rules")
    
    return results

def generate_security_report(audit_results: List[Dict[str, Any]]) -> str:
    """Generate a comprehensive security report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
Firebase Security Audit Report
Generated: {timestamp}
{'=' * 60}

EXECUTIVE SUMMARY
{'-' * 20}
"""
    
    total_issues = sum(len(result["issues"]) for result in audit_results)
    critical_issues = sum(1 for result in audit_results if result["status"] == "fail")
    warning_issues = sum(1 for result in audit_results if result["status"] == "warning")
    
    if total_issues == 0:
        report += "✓ No security issues detected. Your Firebase configuration appears secure.\n"
    else:
        report += f"⚠️  {total_issues} security issues detected:\n"
        report += f"   - {critical_issues} critical issues\n"
        report += f"   - {warning_issues} warnings\n"
    
    report += "\nDETAILED FINDINGS\n"
    report += "-" * 20 + "\n"
    
    for i, result in enumerate(audit_results, 1):
        report += f"\n{i}. Security Check Results:\n"
        report += f"   Status: {result['status'].upper()}\n"
        
        if result["issues"]:
            report += "   Issues:\n"
            for issue in result["issues"]:
                report += f"     - {issue}\n"
        
        if result["recommendations"]:
            report += "   Recommendations:\n"
            for rec in result["recommendations"]:
                report += f"     - {rec}\n"
    
    report += "\nRECOMMENDED ACTIONS\n"
    report += "-" * 20 + "\n"
    
    if critical_issues > 0:
        report += "IMMEDIATE ACTION REQUIRED:\n"
        report += "1. Address all critical security issues before deployment\n"
        report += "2. Review Firebase configuration and credentials\n"
        report += "3. Implement proper environment variable setup\n\n"
    
    if warning_issues > 0:
        report += "RECOMMENDED IMPROVEMENTS:\n"
        report += "1. Address warning-level security issues\n"
        report += "2. Enhance Firestore security rules\n"
        report += "3. Review code for potential security improvements\n\n"
    
    report += "SECURITY BEST PRACTICES:\n"
    report += "1. Use environment variables for all credentials\n"
    report += "2. Never commit service account files to version control\n"
    report += "3. Implement proper Firestore security rules\n"
    report += "4. Regularly audit Firebase access and permissions\n"
    report += "5. Monitor Firebase usage and access logs\n"
    
    return report

def main():
    """Main audit function"""
    print("Firebase Security Audit")
    print("=" * 60)
    print("Performing comprehensive security audit...")
    
    # Run all security checks
    audit_results = []
    
    try:
        audit_results.append(check_environment_variables())
        audit_results.append(check_service_account_files())
        audit_results.append(check_gitignore_rules())
        audit_results.append(check_code_for_hardcoded_secrets())
        audit_results.append(check_firebase_security_rules())
        
        # Generate and save report
        report = generate_security_report(audit_results)
        
        # Save report to file
        report_path = f"firebase_security_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 Security audit report saved to: {report_path}")
        
        # Print summary
        total_issues = sum(len(result["issues"]) for result in audit_results)
        critical_issues = sum(1 for result in audit_results if result["status"] == "fail")
        
        if total_issues == 0:
            print("\n🎉 Security audit completed successfully!")
            print("   No security issues detected.")
        else:
            print(f"\n⚠️  Security audit completed with {total_issues} issues.")
            if critical_issues > 0:
                print(f"   {critical_issues} critical issues require immediate attention.")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Security audit failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())