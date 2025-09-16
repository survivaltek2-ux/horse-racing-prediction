#!/usr/bin/env python3
"""
Firebase Access Pattern Monitor
Monitors and audits Firebase database access patterns for security analysis.
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from firebase_config import db, auth
    FIREBASE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Firebase not available: {e}")
    FIREBASE_AVAILABLE = False

class FirebaseAccessMonitor:
    """Monitor Firebase access patterns and generate security insights."""
    
    def __init__(self):
        self.access_log = []
        self.suspicious_patterns = []
        self.monitoring_start = datetime.now()
        
    def log_access(self, collection, operation, user_id=None, ip_address=None, success=True):
        """Log a database access event."""
        access_event = {
            'timestamp': datetime.now().isoformat(),
            'collection': collection,
            'operation': operation,
            'user_id': user_id,
            'ip_address': ip_address,
            'success': success
        }
        self.access_log.append(access_event)
    
    def analyze_access_patterns(self):
        """Analyze access patterns for suspicious activity."""
        print("Firebase Access Pattern Analysis")
        print("=" * 60)
        
        if not self.access_log:
            print("⚠️  No access events to analyze")
            return
        
        # Analyze by collection
        collection_stats = Counter(event['collection'] for event in self.access_log)
        print("\n1. Collection Access Statistics")
        print("-" * 40)
        for collection, count in collection_stats.most_common():
            print(f"  {collection}: {count} accesses")
        
        # Analyze by operation
        operation_stats = Counter(event['operation'] for event in self.access_log)
        print("\n2. Operation Statistics")
        print("-" * 40)
        for operation, count in operation_stats.most_common():
            print(f"  {operation}: {count} operations")
        
        # Analyze by user
        user_stats = Counter(event['user_id'] for event in self.access_log if event['user_id'])
        print("\n3. User Activity Statistics")
        print("-" * 40)
        for user_id, count in user_stats.most_common(10):
            print(f"  {user_id}: {count} operations")
        
        # Analyze failed operations
        failed_operations = [event for event in self.access_log if not event['success']]
        print(f"\n4. Failed Operations: {len(failed_operations)}")
        print("-" * 40)
        if failed_operations:
            for event in failed_operations[-5:]:  # Show last 5 failures
                print(f"  {event['timestamp']}: {event['operation']} on {event['collection']}")
    
    def detect_suspicious_patterns(self):
        """Detect suspicious access patterns."""
        print("\n5. Suspicious Pattern Detection")
        print("-" * 40)
        
        suspicious_count = 0
        
        # Check for rapid successive requests from same user
        user_requests = defaultdict(list)
        for event in self.access_log:
            if event['user_id']:
                user_requests[event['user_id']].append(event['timestamp'])
        
        for user_id, timestamps in user_requests.items():
            if len(timestamps) > 50:  # More than 50 requests
                self.suspicious_patterns.append({
                    'type': 'High frequency access',
                    'user_id': user_id,
                    'count': len(timestamps),
                    'severity': 'medium'
                })
                suspicious_count += 1
        
        # Check for failed authentication attempts
        failed_auth = [event for event in self.access_log 
                      if not event['success'] and 'auth' in event['operation'].lower()]
        
        if len(failed_auth) > 10:
            self.suspicious_patterns.append({
                'type': 'Multiple authentication failures',
                'count': len(failed_auth),
                'severity': 'high'
            })
            suspicious_count += 1
        
        # Check for admin collection access by non-admin users
        admin_access = [event for event in self.access_log 
                       if event['collection'] == 'api_credentials' and not event['success']]
        
        for event in admin_access:
            if event['user_id']:
                self.suspicious_patterns.append({
                    'type': 'Unauthorized admin collection access attempt',
                    'user_id': event['user_id'],
                    'collection': event['collection'],
                    'severity': 'critical'
                })
                suspicious_count += 1
        
        if suspicious_count == 0:
            print("✓ No suspicious patterns detected")
        else:
            print(f"⚠️  {suspicious_count} suspicious patterns detected:")
            for pattern in self.suspicious_patterns:
                severity_icon = "🔴" if pattern['severity'] == 'critical' else "🟡" if pattern['severity'] == 'high' else "🟠"
                print(f"  {severity_icon} {pattern['type']}")
                if 'user_id' in pattern:
                    print(f"    User: {pattern['user_id']}")
                if 'count' in pattern:
                    print(f"    Count: {pattern['count']}")
    
    def generate_security_recommendations(self):
        """Generate security recommendations based on access patterns."""
        print("\n6. Security Recommendations")
        print("-" * 40)
        
        recommendations = []
        
        # Check collection access patterns
        collection_stats = Counter(event['collection'] for event in self.access_log)
        
        if collection_stats.get('api_credentials', 0) > 0:
            recommendations.append("Monitor API credentials access closely")
        
        if collection_stats.get('users', 0) > collection_stats.get('races', 0) * 2:
            recommendations.append("High user collection access - verify legitimate use")
        
        # Check for failed operations
        failed_operations = [event for event in self.access_log if not event['success']]
        if len(failed_operations) > len(self.access_log) * 0.1:  # More than 10% failures
            recommendations.append("High failure rate - investigate authentication issues")
        
        # General recommendations
        recommendations.extend([
            "Implement rate limiting for API endpoints",
            "Set up real-time monitoring alerts",
            "Regular security rule audits",
            "User access review and cleanup",
            "Implement session timeout policies"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    def simulate_access_patterns(self):
        """Simulate various access patterns for testing."""
        print("Simulating access patterns for analysis...")
        
        # Simulate normal user activity
        users = ['user_001', 'user_002', 'user_003', 'admin_001']
        collections = ['races', 'users', 'predictions', 'api_credentials']
        operations = ['read', 'write', 'update', 'delete']
        
        # Normal access patterns
        for i in range(100):
            user = users[i % len(users)]
            collection = collections[i % len(collections)]
            operation = operations[i % len(operations)]
            
            # Admin-only collections
            if collection == 'api_credentials' and not user.endswith('_admin'):
                success = False
            else:
                success = True
            
            self.log_access(collection, operation, user, f"192.168.1.{i%10}", success)
        
        # Simulate suspicious patterns
        # High frequency access
        for i in range(60):
            self.log_access('users', 'read', 'suspicious_user', '192.168.1.100', True)
        
        # Failed authentication attempts
        for i in range(15):
            self.log_access('users', 'auth_login', 'attacker', '192.168.1.200', False)
        
        print(f"✓ Simulated {len(self.access_log)} access events")
    
    def export_monitoring_report(self):
        """Export monitoring report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"firebase_access_monitor_report_{timestamp}.json"
        
        report = {
            'monitoring_period': {
                'start': self.monitoring_start.isoformat(),
                'end': datetime.now().isoformat(),
                'duration_minutes': (datetime.now() - self.monitoring_start).total_seconds() / 60
            },
            'summary': {
                'total_events': len(self.access_log),
                'unique_users': len(set(event['user_id'] for event in self.access_log if event['user_id'])),
                'unique_collections': len(set(event['collection'] for event in self.access_log)),
                'failed_operations': len([e for e in self.access_log if not e['success']]),
                'suspicious_patterns': len(self.suspicious_patterns)
            },
            'access_log': self.access_log,
            'suspicious_patterns': self.suspicious_patterns,
            'recommendations': [
                "Implement rate limiting for API endpoints",
                "Set up real-time monitoring alerts",
                "Regular security rule audits",
                "User access review and cleanup",
                "Implement session timeout policies"
            ]
        }
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n📄 Access monitoring report saved to: {report_file}")
        except Exception as e:
            print(f"⚠️  Error saving report: {e}")
        
        return report
    
    def run_monitoring_analysis(self):
        """Run complete monitoring analysis."""
        print("Firebase Access Pattern Monitoring")
        print("=" * 60)
        print(f"Analysis started at {datetime.now()}")
        
        # Simulate access patterns for demonstration
        self.simulate_access_patterns()
        
        # Analyze patterns
        self.analyze_access_patterns()
        
        # Detect suspicious activity
        self.detect_suspicious_patterns()
        
        # Generate recommendations
        self.generate_security_recommendations()
        
        # Export report
        report = self.export_monitoring_report()
        
        print("\n" + "=" * 60)
        print("MONITORING SUMMARY")
        print("=" * 60)
        print(f"Total Events: {report['summary']['total_events']}")
        print(f"Unique Users: {report['summary']['unique_users']}")
        print(f"Failed Operations: {report['summary']['failed_operations']}")
        print(f"Suspicious Patterns: {report['summary']['suspicious_patterns']}")
        
        if report['summary']['suspicious_patterns'] > 0:
            print("\n⚠️  Suspicious patterns detected. Review the report for details.")
            return False
        else:
            print("\n✓ No critical security issues detected.")
            return True

def main():
    """Main function to run Firebase access monitoring."""
    monitor = FirebaseAccessMonitor()
    success = monitor.run_monitoring_analysis()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()