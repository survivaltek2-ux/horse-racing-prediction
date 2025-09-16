#!/usr/bin/env python3
"""
Comprehensive API Race Data Test Script
Tests all API endpoints for pulling race information from different providers.
"""

import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
PROVIDERS = ["mock", "sample", "theracingapi"]

def test_api_connection(provider):
    """Test API connection for a specific provider."""
    print(f"\n🔗 Testing connection to {provider}...")
    
    try:
        start_time = time.time()
        response = requests.get(f"{BASE_URL}/api/test_connection/{provider}")
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Connection successful ({end_time - start_time:.3f}s)")
            print(f"   Provider: {data.get('provider')}")
            print(f"   Message: {data.get('message')}")
            print(f"   Races found: {data.get('races_found', 'N/A')}")
            return True
        else:
            print(f"❌ Connection failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        return False

def test_race_data_retrieval(provider, days_ahead=7):
    """Test race data retrieval for a specific provider."""
    print(f"\n📊 Testing race data retrieval from {provider}...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/fetch_races",
            headers={"Content-Type": "application/json"},
            json={"provider": provider, "days_ahead": days_ahead}
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Data retrieval successful ({end_time - start_time:.3f}s)")
            print(f"   Success: {data.get('success')}")
            print(f"   Race count: {data.get('count', 0)}")
            
            races = data.get('races', [])
            if races:
                print(f"   Sample race: {races[0].get('name', 'Unknown')}")
                print(f"   Date: {races[0].get('date', 'Unknown')}")
                print(f"   Location: {races[0].get('location', 'Unknown')}")
                print(f"   Distance: {races[0].get('distance', 'Unknown')}")
                print(f"   Purse: ${races[0].get('purse', 0):,.2f}")
            
            return True, len(races)
        else:
            print(f"❌ Data retrieval failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False, 0
            
    except Exception as e:
        print(f"❌ Data retrieval error: {str(e)}")
        return False, 0

def test_providers_endpoint():
    """Test the providers endpoint to see available APIs."""
    print(f"\n🔍 Testing providers endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/providers")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Providers endpoint successful")
            print(f"   Available providers: {len(data)}")
            
            for provider_name, provider_info in data.items():
                configured = provider_info.get('configured', False)
                status = "✅ Configured" if configured else "⚠️  Not configured"
                print(f"   - {provider_name}: {status}")
                print(f"     Name: {provider_info.get('name', 'Unknown')}")
                print(f"     Features: {', '.join(provider_info.get('features', []))}")
                
            return True
        else:
            print(f"❌ Providers endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Providers endpoint error: {str(e)}")
        return False

def test_import_races(provider):
    """Test importing races into the database."""
    print(f"\n💾 Testing race import from {provider}...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/import_races",
            headers={"Content-Type": "application/json"},
            json={"provider": provider, "days_ahead": 3}
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Race import successful ({end_time - start_time:.3f}s)")
            print(f"   Success: {data.get('success')}")
            print(f"   Message: {data.get('message', 'No message')}")
            print(f"   Imported: {data.get('imported', 0)} races")
            print(f"   Skipped: {data.get('skipped', 0)} races")
            return True
        else:
            print(f"❌ Race import failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Race import error: {str(e)}")
        return False

def main():
    """Run comprehensive API tests."""
    print("🏇 Horse Racing API Test Suite")
    print("=" * 50)
    print(f"Testing against: {BASE_URL}")
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test providers endpoint
    providers_ok = test_providers_endpoint()
    
    # Test each provider
    results = {}
    for provider in PROVIDERS:
        print(f"\n{'=' * 30}")
        print(f"Testing Provider: {provider.upper()}")
        print(f"{'=' * 30}")
        
        # Test connection
        connection_ok = test_api_connection(provider)
        
        # Test race data retrieval
        data_ok, race_count = test_race_data_retrieval(provider)
        
        # Test race import (only if data retrieval works)
        import_ok = False
        if data_ok and race_count > 0:
            import_ok = test_import_races(provider)
        
        results[provider] = {
            'connection': connection_ok,
            'data_retrieval': data_ok,
            'race_count': race_count,
            'import': import_ok
        }
    
    # Summary
    print(f"\n{'=' * 50}")
    print("📋 TEST SUMMARY")
    print(f"{'=' * 50}")
    
    print(f"Providers endpoint: {'✅ PASS' if providers_ok else '❌ FAIL'}")
    
    for provider, result in results.items():
        print(f"\n{provider.upper()}:")
        print(f"  Connection: {'✅ PASS' if result['connection'] else '❌ FAIL'}")
        print(f"  Data retrieval: {'✅ PASS' if result['data_retrieval'] else '❌ FAIL'}")
        print(f"  Race count: {result['race_count']}")
        print(f"  Import: {'✅ PASS' if result['import'] else '❌ FAIL'}")
    
    # Recommendations
    print(f"\n📝 RECOMMENDATIONS:")
    
    working_providers = [p for p, r in results.items() if r['data_retrieval'] and r['race_count'] > 0]
    if working_providers:
        print(f"✅ Working providers for race data: {', '.join(working_providers)}")
    else:
        print("⚠️  No providers are returning race data")
    
    configured_providers = []
    not_configured_providers = []
    
    # This would need to be determined from the providers endpoint response
    print("💡 To get real race data:")
    print("   1. Configure API credentials for theracingapi or sample providers")
    print("   2. Use the mock provider for testing and development")
    print("   3. Check the admin panel at /admin/api_credentials for configuration")
    
    print(f"\nTest completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()