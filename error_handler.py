"""
Error handling module for the Flask application.
Provides comprehensive error handling, exception catching, form validation, and database operation logging.
"""

import logging
import functools
import traceback
from flask import request, jsonify, render_template, current_app
from datetime import datetime


class ErrorHandler:
    """Comprehensive error handler for the Flask application"""
    
    def __init__(self):
        self.logger = logging.getLogger('error_handler')
        self.logger.setLevel(logging.ERROR)
        
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.ERROR)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        if not self.logger.handlers:
            self.logger.addHandler(handler)
    
    def init_app(self, app):
        """Initialize error handling with the Flask app"""
        self.app = app
        
        # Register error handlers
        app.register_error_handler(404, self.handle_404)
        app.register_error_handler(403, self.handle_403)
        app.register_error_handler(500, self.handle_500)
        app.register_error_handler(Exception, self.handle_generic_exception)
        
        app.logger.info("Error handler initialized")
    
    def handle_404(self, error):
        """Handle 404 Not Found errors"""
        self.logger.error(f"404 error: {request.url}")
        
        # Check if this is an API request
        if request.path.startswith('/api/') or request.is_json:
            return jsonify({
                'error': 'Not Found',
                'message': 'The requested resource was not found',
                'status_code': 404
            }), 404
        
        return render_template('errors/404.html'), 404
    
    def handle_403(self, error):
        """Handle 403 Forbidden errors"""
        self.logger.error(f"403 error: {request.url}")
        
        # Check if this is an API request
        if request.path.startswith('/api/') or request.is_json:
            return jsonify({
                'error': 'Forbidden',
                'message': 'You do not have permission to access this resource',
                'status_code': 403
            }), 403
        
        return render_template('errors/403.html'), 403
    
    def handle_500(self, error):
        """Handle 500 Internal Server Error"""
        self.logger.error(f"500 error: {str(error)}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Check if this is an API request
        if request.path.startswith('/api/') or request.is_json:
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'An internal server error occurred',
                'status_code': 500
            }), 500
        
        return render_template('errors/500.html'), 500
    
    def handle_generic_exception(self, error):
        """Handle any unhandled exceptions"""
        self.logger.error(f"Unhandled exception: {str(error)}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Check if this is an API request
        if request.path.startswith('/api/') or request.is_json:
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred',
                'status_code': 500
            }), 500
        
        return render_template('errors/500.html'), 500


def catch_exceptions(f):
    """Decorator to catch and handle exceptions in route functions"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            error_handler.logger.error(f"Exception in {f.__name__}: {str(e)}")
            error_handler.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Check if this is an API request
            if request.path.startswith('/api/') or request.is_json:
                return jsonify({
                    'error': 'Internal Server Error',
                    'message': f'Error in {f.__name__}: {str(e)}',
                    'status_code': 500
                }), 500
            
            return render_template('errors/500.html'), 500
    
    return wrapper


def validate_form_data(required_fields):
    """Decorator to validate form data"""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            missing_fields = []
            
            for field in required_fields:
                if field not in request.form or not request.form[field].strip():
                    missing_fields.append(field)
            
            if missing_fields:
                error_message = f"Missing required fields: {', '.join(missing_fields)}"
                error_handler.logger.warning(f"Form validation failed: {error_message}")
                
                # Check if this is an API request
                if request.path.startswith('/api/') or request.is_json:
                    return jsonify({
                        'error': 'Validation Error',
                        'message': error_message,
                        'missing_fields': missing_fields,
                        'status_code': 400
                    }), 400
                
                # For web requests, flash the error and redirect
                from flask import flash, redirect, url_for
                flash(error_message, 'error')
                return redirect(request.referrer or url_for('index'))
            
            return f(*args, **kwargs)
        
        return wrapper
    return decorator


def log_database_operation(operation_type):
    """Decorator to log database operations"""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            
            try:
                result = f(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                error_handler.logger.info(
                    f"Database {operation_type} in {f.__name__} completed in {duration:.4f} seconds"
                )
                
                return result
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                error_handler.logger.error(
                    f"Database {operation_type} in {f.__name__} failed after {duration:.4f} seconds: {str(e)}"
                )
                raise
        
        return wrapper
    return decorator


# Create global instance
error_handler = ErrorHandler()