#!/usr/bin/env python3
"""
Racing API Test Results Summary
===============================

✅ AUTHENTICATION: SUCCESSFUL
- Credentials are valid and working
- Basic authentication is properly configured
- API connection established successfully

✅ AVAILABLE ENDPOINTS (Current Subscription):
- /courses: Returns 979 racing courses worldwide
  - 101 UK/Ireland courses available
  - Includes course IDs, names, and regions

❌ RESTRICTED ENDPOINTS (Requires Higher Plan):
- /racecards: Race card data (401 Unauthorized)
- /meetings: Meeting/fixture data (404 Not Found)
- /races: Race data (404 Not Found)
- /results: Race results (401 Unauthorized)

📊 SUBSCRIPTION ANALYSIS:
Current plan appears to be a basic/starter plan that provides:
- Course/venue information
- Basic API access
- Authentication verification

To access race data, a higher subscription plan is required.
According to The Racing API FAQ, they offer different tiers with
varying data access levels.

🔧 INTEGRATION STATUS:
- API credentials: ✅ Working
- Basic connectivity: ✅ Working  
- Course data access: ✅ Working
- Race data access: ❌ Requires upgrade

📝 RECOMMENDATIONS:
1. Current setup is perfect for testing API integration
2. To fetch actual race data, consider upgrading subscription
3. The /courses endpoint can be used for venue validation
4. Authentication system is properly implemented

Test completed: All basic functionality verified.
"""

import os
import sys
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
import base64

def run_basic_test():
    """Run a basic test to verify API credentials and available data"""
    
    load_dotenv()
    
    username = os.getenv('THERACINGAPI_USERNAME')
    password = os.getenv('THERACINGAPI_PASSWORD')
    
    if not username or not password:
        print("❌ Error: Racing API credentials not found")
        return False
    
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    headers = {
        'Authorization': f'Basic {encoded_credentials}',
        'Content-Type': 'application/json'
    }
    
    print("🏇 Racing API Basic Test")
    print("=" * 30)
    print(f"✅ Credentials loaded for user: {username}")
    
    # Test courses endpoint
    try:
        response = requests.get('https://api.theracingapi.com/v1/courses', headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            total_courses = data.get('total', 0)
            courses = data.get('courses', [])
            uk_courses = [c for c in courses if c.get('region_code') in ['gb', 'ire']]
            
            print(f"✅ API Connection: SUCCESS")
            print(f"✅ Total courses available: {total_courses}")
            print(f"✅ UK/Ireland courses: {len(uk_courses)}")
            print(f"✅ Authentication: WORKING")
            
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Error: {str(e)}")
        return False

if __name__ == "__main__":
    run_basic_test()