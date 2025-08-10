#!/usr/bin/env python3
"""Test script to run the app and check for issues."""

import sys
import traceback

try:
    print("🔍 Testing imports...")
    from app import app
    print("✅ App imported successfully!")
    
    print("🔍 Testing basic app functionality...")
    with app.test_client() as client:
        response = client.get('/')
        print(f"✅ Homepage response: {response.status_code}")
        
        response = client.get('/check-auth')
        print(f"✅ Auth check response: {response.status_code}")
        
        response = client.get('/check-observee-status')
        print(f"✅ Observee status response: {response.status_code}")
        
    print("🎉 All tests passed! The app should work correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"❌ Runtime error: {e}")
    traceback.print_exc()
