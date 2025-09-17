"""
Enhanced Error Handling Module for Horse Racing Prediction System
Provides comprehensive error handling, logging, and debugging capabilities
"""

import logging
import traceback
import sys
from functools import wraps
from flask import request, jsonify, render_template, flash, redirect, url_for
from werkzeug.exceptions import HTTPException
import json
from datetime import datetime

# Get logger from logging config
logger = logging.getLogger('debug')

class ErrorHandler:
    """Comprehensive error handling system"""
    
    def __init__(self, app=None):
        self.app = app
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize error handling for Flask app"""
        self.app = app
        
        # Register error handlers
        app.errorhandler(404)(self.handle_404)
        app.errorhandler(500)(self.handle_500)
        app.errorhandler(403)(self.handle_403)
        app.errorhandler(Exception)(self.handle_generic_exception)
        
        # Add before/after request handlers
        app.before_request(self.log_request_info)
        app.after_request(self.log_response_info)
        
        logger.info("Error handling system initialized")
    
    def log_request_info(self):
        """Log detailed request information"""
        try:
            request_data = {
                'method': request.method,
                'url': request.url,
                'endpoint': request.endpoint,
                'remote_addr': request.remote_addr,
                'user_agent': str(request.user_agent),
                'timestamp': datetime.now().isoformat()
            }
            
            # Log form data for POST requests (excluding sensitive data)
            if request.method == 'POST' and request.form:
                form_data = {}
                for key, value in request.form.items():
                    if 'password' not in key.lower() and 'secret' not in key.lower():
                        form_data[key] = value
                request_data['form_data'] = form_data
            
            logger.debug(f"Request: {json.dumps(request_data, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error logging request info: {str(e)}")
    
    def log_response_info(self, response):
        """Log response information"""
        try:
            response_data = {
                'status_code': response.status_code,
                'content_type': response.content_type,
                'content_length': response.content_length,
                'endpoint': request.endpoint,
                'timestamp': datetime.now().isoformat()
            }
            
            if response.status_code >= 400:
                logger.warning(f"Error Response: {json.dumps(response_data, indent=2)}")
            else:
                logger.debug(f"Response: {json.dumps(response_data, indent=2)}")
                
        except Exception as e:
            logger.error(f"Error logging response info: {str(e)}")
        
        return response
    
    def handle_404(self, error):
        """Handle 404 errors"""
        error_info = {
            'error_type': '404 Not Found',
            'url': request.url,
            'method': request.method,
            'timestamp': datetime.now().isoformat(),
            'user_agent': str(request.user_agent)
        }
        
        logger.warning(f"404 Error: {json.dumps(error_info, indent=2)}")
        
        if request.is_json:
            return jsonify({'error': 'Page not found', 'status': 404}), 404
        
        flash('The page you requested could not be found.', 'warning')
        return render_template('errors/404.html'), 404
    
    def handle_403(self, error):
        """Handle 403 errors"""
        error_info = {
            'error_type': '403 Forbidden',
            'url': request.url,
            'method': request.method,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.warning(f"403 Error: {json.dumps(error_info, indent=2)}")
        
        if request.is_json:
            return jsonify({'error': 'Access forbidden', 'status': 403}), 403
        
        flash('You do not have permission to access this resource.', 'danger')
        return redirect(url_for('login'))
    
    def handle_500(self, error):
        """Handle 500 errors"""
        error_info = {
            'error_type': '500 Internal Server Error',
            'url': request.url,
            'method': request.method,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.error(f"500 Error: {json.dumps(error_info, indent=2)}")
        
        if request.is_json:
            return jsonify({'error': 'Internal server error', 'status': 500}), 500
        
        flash('An internal server error occurred. Please try again later.', 'danger')
        return render_template('errors/500.html'), 500
    
    def handle_generic_exception(self, error):
        """Handle all other exceptions"""
        if isinstance(error, HTTPException):
            return error
        
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'url': request.url,
            'method': request.method,
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.error(f"Unhandled Exception: {json.dumps(error_info, indent=2)}")
        
        if request.is_json:
            return jsonify({'error': 'An unexpected error occurred', 'status': 500}), 500
        
        flash('An unexpected error occurred. Please try again.', 'danger')
        return render_template('errors/500.html'), 500


def catch_exceptions(f):
    """Decorator to catch and log exceptions in route functions"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            error_info = {
                'function': f.__name__,
                'args': str(args),
                'kwargs': str(kwargs),
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.error(f"Function Exception: {json.dumps(error_info, indent=2)}")
            
            # Re-raise the exception to be handled by Flask's error handlers
            raise
    
    return decorated_function


def validate_form_data(form, required_fields=None):
    """Validate form data and log any issues"""
    if required_fields is None:
        required_fields = []
    
    validation_info = {
        'form_valid': form.validate(),
        'form_errors': form.errors,
        'required_fields': required_fields,
        'timestamp': datetime.now().isoformat()
    }
    
    # Check for missing required fields
    missing_fields = []
    for field in required_fields:
        if hasattr(form, field):
            field_obj = getattr(form, field)
            if not field_obj.data:
                missing_fields.append(field)
    
    validation_info['missing_required_fields'] = missing_fields
    
    if not form.validate() or missing_fields:
        logger.warning(f"Form Validation Failed: {json.dumps(validation_info, indent=2)}")
        return False, validation_info
    else:
        logger.debug(f"Form Validation Passed: {json.dumps(validation_info, indent=2)}")
        return True, validation_info


def log_database_operation(operation, table, data=None, result=None, error=None):
    """Log database operations for debugging"""
    db_info = {
        'operation': operation,
        'table': table,
        'timestamp': datetime.now().isoformat()
    }
    
    if data:
        db_info['data'] = str(data)
    
    if result:
        db_info['result'] = str(result)
    
    if error:
        db_info['error'] = str(error)
        db_info['traceback'] = traceback.format_exc()
        logger.error(f"Database Error: {json.dumps(db_info, indent=2)}")
    else:
        logger.debug(f"Database Operation: {json.dumps(db_info, indent=2)}")


# Initialize error handler instance
error_handler = ErrorHandler()