#!/usr/bin/env python3
"""
Simple debug server script to test the edit_race route
"""

import os
import sys

# Set environment variables
os.environ['DEBUG'] = 'True'
os.environ['FLASK_ENV'] = 'development'

# Import and run the app
from app import app

if __name__ == '__main__':
    print("Starting Flask development server with debug mode...")
    print("Server will be available at: http://localhost:8001/")
    print("Press Ctrl+C to stop the server")
    
    app.run(
        host='0.0.0.0',
        port=8001,
        debug=True,
        use_reloader=False  # Disable reloader to avoid issues
    )