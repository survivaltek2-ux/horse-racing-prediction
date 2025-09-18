"""
Logging configuration module for the Flask application.
Provides debug logging, performance monitoring, and route debugging functionality.
"""

import logging
import time
import functools
from flask import request, g
from datetime import datetime


class DebugLogger:
    """Debug logger for the application"""
    
    def __init__(self):
        self.logger = logging.getLogger('debug_logger')
        self.logger.setLevel(logging.DEBUG)
        
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        if not self.logger.handlers:
            self.logger.addHandler(handler)
    
    def init_app(self, app):
        """Initialize the debug logger with the Flask app"""
        self.app = app
        app.logger.info("Debug logger initialized")
    
    def debug(self, message):
        """Log a debug message"""
        self.logger.debug(message)
    
    def info(self, message):
        """Log an info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log a warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log an error message"""
        self.logger.error(message)


def performance_monitor(f):
    """Decorator to monitor performance of functions"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = f(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Log performance metrics
            debug_logger.info(f"Function {f.__name__} executed in {execution_time:.4f} seconds")
            
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            debug_logger.error(f"Function {f.__name__} failed after {execution_time:.4f} seconds: {str(e)}")
            raise
    
    return wrapper


def debug_route(f):
    """Decorator to debug route requests"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        # Log request details
        debug_logger.info(f"Route {request.endpoint} called")
        debug_logger.debug(f"Request method: {request.method}")
        debug_logger.debug(f"Request URL: {request.url}")
        debug_logger.debug(f"Request args: {dict(request.args)}")
        
        if request.method in ['POST', 'PUT', 'PATCH']:
            debug_logger.debug(f"Request form data: {dict(request.form)}")
        
        start_time = time.time()
        try:
            result = f(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            debug_logger.info(f"Route {request.endpoint} completed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            debug_logger.error(f"Route {request.endpoint} failed after {execution_time:.4f} seconds: {str(e)}")
            raise
    
    return wrapper


# Create global instance
debug_logger = DebugLogger()