#!/usr/bin/env python3
"""
Train AI Models via Flask API

This script authenticates with the Flask app and calls the AI training endpoint.
"""

import requests
import json
import sys

def train_ai_models():
    """Train AI models via the Flask API"""
    base_url = "http://127.0.0.1:8000"
    
    # Create a session to maintain cookies
    session = requests.Session()
    
    try:
        # First, try to access the training endpoint directly
        print("🤖 Attempting to train AI models...")
        
        # Login first
        login_data = {
            'username': 'admin',
            'password': 'admin123'
        }
        
        print("🔐 Logging in...")
        login_response = session.post(f"{base_url}/login", data=login_data)
        
        if login_response.status_code == 200:
            print("✅ Login successful")
        else:
            print(f"❌ Login failed: {login_response.status_code}")
            return False
        
        # Now call the AI training endpoint
        print("🚀 Calling AI training endpoint...")
        train_response = session.post(f"{base_url}/api/train_ai")
        
        if train_response.status_code == 200:
            result = train_response.json()
            print("✅ AI training API call successful")
            print(f"   Success: {result.get('success', False)}")
            print(f"   Message: {result.get('message', 'No message')}")
            print(f"   Timestamp: {result.get('timestamp', 'No timestamp')}")
            return result.get('success', False)
        else:
            print(f"❌ AI training failed: {train_response.status_code}")
            print(f"   Response: {train_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error training AI models: {str(e)}")
        return False

if __name__ == "__main__":
    success = train_ai_models()
    if success:
        print("\n🎉 AI models trained successfully!")
        sys.exit(0)
    else:
        print("\n💥 AI model training failed!")
        sys.exit(1)