#!/usr/bin/env python3
"""
Web AI Verification Test
This script tests the web interface to show how to verify AI usage through the browser.
"""

import requests
import json
import sys
import time

def test_web_ai_endpoints():
    """Test AI endpoints through web interface"""
    base_url = "http://localhost:5000"
    
    print("🌐 TESTING WEB AI ENDPOINTS")
    print("-" * 40)
    
    # Test if server is running
    try:
        response = requests.get(base_url, timeout=5)
        print(f"✅ Server is running (Status: {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"❌ Server not accessible: {e}")
        print("💡 Make sure to run: python3 app.py")
        return False
    
    # Test AI prediction endpoint
    try:
        # Test with a sample race ID
        ai_url = f"{base_url}/api/predict_ai"
        
        # Sample data for AI prediction
        test_data = {
            "race_id": 1,
            "track_condition": "good",
            "weather": "clear",
            "distance": 1200
        }
        
        print(f"\n🎯 Testing AI API endpoint: {ai_url}")
        response = requests.post(ai_url, json=test_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ AI API endpoint working")
            
            # Check for AI indicators in response
            ai_indicators = []
            
            if 'algorithm' in result:
                algorithm = result['algorithm']
                print(f"   Algorithm: {algorithm}")
                if 'ai' in algorithm.lower() or 'ensemble' in algorithm.lower():
                    ai_indicators.append('AI Algorithm')
            
            if 'ai_insights' in result:
                insights = result['ai_insights']
                print(f"   AI Insights: {len(insights) if insights else 0} items")
                if insights:
                    ai_indicators.append('AI Insights')
                    
                    # Check if using fallback
                    fallback_terms = ['fallback', 'not trained', 'not available']
                    is_fallback = any(
                        any(term in str(insight).lower() for term in fallback_terms)
                        for insight in insights
                    )
                    
                    if is_fallback:
                        print("   ⚠️  AI using fallback (models not trained)")
                    else:
                        print("   ✅ AI using trained models")
            
            if 'confidence_scores' in result:
                scores = result['confidence_scores']
                print(f"   Confidence Scores: {scores}")
                if 'ai_confidence' in scores:
                    ai_indicators.append('AI Confidence')
            
            print(f"\n📊 AI Indicators Found: {', '.join(ai_indicators) if ai_indicators else 'None'}")
            return len(ai_indicators) > 0
            
        else:
            print(f"❌ AI API failed (Status: {response.status_code})")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error testing AI endpoint: {e}")
        return False

def show_browser_verification_steps():
    """Show users how to verify AI in the browser"""
    print("\n🖥️  BROWSER VERIFICATION STEPS")
    print("=" * 50)
    
    print("\n1. 🏠 OPEN THE HOME PAGE:")
    print("   • Go to: http://localhost:5000")
    print("   • Look for 'AI Prediction' or 'Enhanced Prediction' buttons")
    
    print("\n2. 🎯 USE AI PREDICTION:")
    print("   • Click 'AI Prediction' or go to: http://localhost:5000/predict_ai/1")
    print("   • Compare with regular prediction: http://localhost:5000/predict/1")
    
    print("\n3. 🔍 CHECK THE RESULTS PAGE:")
    print("   • Look for 'AI Enabled' badge at the top")
    print("   • Check algorithm shows 'AI Ensemble' or similar")
    print("   • Look for 'AI Confidence' in confidence scores")
    print("   • Check insights mention 'neural networks' or 'deep learning'")
    
    print("\n4. 🔬 COMPARE PREDICTIONS:")
    print("   • Traditional: Shows 'Random Forest' algorithm")
    print("   • AI: Shows 'AI Ensemble' algorithm")
    print("   • AI has additional confidence breakdowns")
    print("   • AI insights are more detailed")
    
    print("\n5. 🚨 FALLBACK INDICATORS:")
    print("   • If AI models aren't trained, insights will say 'fallback'")
    print("   • Algorithm might still show 'AI Ensemble' but with fallback note")
    print("   • This is normal for new installations")

def main():
    """Run web AI verification"""
    print("🤖 WEB AI VERIFICATION")
    print("=" * 50)
    
    # Test web endpoints
    web_working = test_web_ai_endpoints()
    
    # Show browser steps
    show_browser_verification_steps()
    
    # Summary
    print(f"\n{'='*50}")
    print("WEB VERIFICATION SUMMARY")
    print(f"{'='*50}")
    
    if web_working:
        print("✅ AI web endpoints are working")
        print("🌐 You can verify AI usage through the browser")
    else:
        print("❌ AI web endpoints need attention")
        print("🔧 Check server status and try again")
    
    print("\n💡 QUICK TEST:")
    print("   1. Open: http://localhost:5000/predict_ai/1")
    print("   2. Look for 'AI Enabled' badge")
    print("   3. Check algorithm type in results")
    print("   4. Compare with: http://localhost:5000/predict/1")
    
    return web_working

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)