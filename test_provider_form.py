#!/usr/bin/env python3
"""
Test script to validate the new provider selection functionality
"""

from app import app
from forms import APICredentialsForm

def test_provider_form():
    """Test the APICredentialsForm with new provider selection"""
    
    with app.test_request_context():
        print("Testing APICredentialsForm with new provider selection...")
        
        # Test 1: Check if provider choices are available
        form = APICredentialsForm()
        print(f"\nProvider choices: {form.provider.choices}")
        
        # Test 2: Test with predefined provider
        form_data = {
            'provider': 'theracingapi',
            'username': 'testuser',
            'password': 'testpass',
            'description': 'Test Racing API credentials',
            'is_active': True
        }
        
        form = APICredentialsForm(data=form_data)
        print(f"\nTest with theracingapi provider:")
        print(f"Form valid: {form.validate()}")
        if not form.validate():
            print(f"Form errors: {form.errors}")
        
        # Test 3: Test with custom provider
        form_data_custom = {
            'provider': 'custom',
            'custom_provider_name': 'my_custom_api',
            'api_key': 'test_api_key_123',
            'description': 'Custom API provider',
            'is_active': True
        }
        
        form_custom = APICredentialsForm(data=form_data_custom)
        print(f"\nTest with custom provider:")
        print(f"Form valid: {form_custom.validate()}")
        if not form_custom.validate():
            print(f"Form errors: {form_custom.errors}")
        
        # Test 4: Test validation without required fields
        form_empty = APICredentialsForm(data={'provider': ''})
        print(f"\nTest with empty provider:")
        print(f"Form valid: {form_empty.validate()}")
        if not form_empty.validate():
            print(f"Form errors: {form_empty.errors}")
        
        print("\n✅ Provider form testing completed!")

if __name__ == '__main__':
    test_provider_form()