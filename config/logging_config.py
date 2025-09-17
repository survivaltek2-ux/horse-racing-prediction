"""
Comprehensive Logging Configuration for Horse Racing Prediction System
Provides detailed debugging, error tracking, and performance monitoring
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
import json
import traceback
from functools import wraps
import time


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        # Format the message
        formatted = super().format(record)
        return formatted


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class DebugLogger:
    """Enhanced debug logger with performance tracking and detailed error reporting"""
    
    def __init__(self, app=None):
        self.app = app
        self.loggers = {}
        self.performance_data = {}
        
    def init_app(self, app):
        """Initialize logging for Flask app"""
        self.app = app
        
        # Create logs directory
        log_dir = Path(app.root_path) / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging levels
        debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
        log_level = logging.DEBUG if debug_mode else logging.INFO
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler for general logs
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'app.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # JSON handler for structured logs
        json_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'app.json',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        json_handler.setLevel(log_level)
        json_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(json_handler)
        
        # Error handler for critical issues
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'errors.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=10
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
        
        # Performance handler
        perf_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'performance.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=5
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(file_formatter)
        
        # Create performance logger
        perf_logger = logging.getLogger('performance')
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
        
        # Configure Flask app logger
        app.logger.setLevel(log_level)
        
        # Configure specific loggers
        self._configure_specific_loggers(log_level)
        
        # Add request logging
        self._setup_request_logging(app)
        
        app.logger.info("Comprehensive logging system initialized")
        
    def _configure_specific_loggers(self, log_level):
        """Configure specific loggers for different components"""
        
        # Database logger
        db_logger = logging.getLogger('database')
        db_logger.setLevel(log_level)
        
        # API logger
        api_logger = logging.getLogger('api')
        api_logger.setLevel(log_level)
        
        # Prediction logger
        prediction_logger = logging.getLogger('prediction')
        prediction_logger.setLevel(log_level)
        
        # Security logger
        security_logger = logging.getLogger('security')
        security_logger.setLevel(logging.WARNING)
        
        # Store loggers
        self.loggers = {
            'database': db_logger,
            'api': api_logger,
            'prediction': prediction_logger,
            'security': security_logger,
            'performance': logging.getLogger('performance')
        }
    
    def _setup_request_logging(self, app):
        """Setup detailed request logging"""
        
        @app.before_request
        def log_request_info():
            """Log incoming request details"""
            from flask import request, g
            
            g.start_time = time.time()
            
            # Log request details
            app.logger.debug(f"Request: {request.method} {request.url}")
            app.logger.debug(f"Headers: {dict(request.headers)}")
            app.logger.debug(f"Remote Address: {request.remote_addr}")
            app.logger.debug(f"User Agent: {request.user_agent}")
            
            if request.is_json:
                app.logger.debug(f"JSON Data: {request.get_json()}")
            elif request.form:
                # Don't log sensitive form data
                safe_form = {k: v for k, v in request.form.items() 
                           if 'password' not in k.lower() and 'secret' not in k.lower()}
                app.logger.debug(f"Form Data: {safe_form}")
        
        @app.after_request
        def log_response_info(response):
            """Log response details and performance"""
            from flask import g, request
            
            # Calculate request duration
            duration = time.time() - getattr(g, 'start_time', time.time())
            
            # Log response details
            app.logger.debug(f"Response: {response.status_code} - {duration:.3f}s")
            
            # Log performance data
            perf_logger = self.loggers['performance']
            perf_logger.info(f"REQUEST_PERFORMANCE: {request.method} {request.endpoint} "
                           f"- {response.status_code} - {duration:.3f}s")
            
            # Track slow requests
            if duration > 2.0:  # Requests taking more than 2 seconds
                app.logger.warning(f"SLOW_REQUEST: {request.method} {request.url} "
                                 f"took {duration:.3f}s")
            
            return response
        
        @app.errorhandler(Exception)
        def log_exception(error):
            """Log unhandled exceptions"""
            app.logger.error(f"Unhandled exception: {error}", exc_info=True)
            
            # Log to security logger if it's a potential security issue
            if any(keyword in str(error).lower() for keyword in 
                   ['unauthorized', 'forbidden', 'access denied', 'permission']):
                self.loggers['security'].warning(f"Security-related error: {error}", 
                                                exc_info=True)
            
            # Return a generic error response
            from flask import jsonify
            return jsonify({'error': 'Internal server error'}), 500
    
    def get_logger(self, name):
        """Get a specific logger"""
        return self.loggers.get(name, logging.getLogger(name))
    
    def log_database_operation(self, operation, table, details=None):
        """Log database operations"""
        db_logger = self.loggers['database']
        message = f"DB_OPERATION: {operation} on {table}"
        if details:
            message += f" - {details}"
        db_logger.info(message)
    
    def log_api_call(self, endpoint, method, status_code, duration=None):
        """Log API calls"""
        api_logger = self.loggers['api']
        message = f"API_CALL: {method} {endpoint} - {status_code}"
        if duration:
            message += f" - {duration:.3f}s"
        api_logger.info(message)
    
    def log_prediction_event(self, event_type, race_id, details=None):
        """Log prediction-related events"""
        pred_logger = self.loggers['prediction']
        message = f"PREDICTION_{event_type}: Race {race_id}"
        if details:
            message += f" - {details}"
        pred_logger.info(message)
    
    def log_security_event(self, event_type, user_id=None, details=None):
        """Log security-related events"""
        sec_logger = self.loggers['security']
        message = f"SECURITY_{event_type}"
        if user_id:
            message += f": User {user_id}"
        if details:
            message += f" - {details}"
        sec_logger.warning(message)


def performance_monitor(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Log performance
            perf_logger = logging.getLogger('performance')
            perf_logger.info(f"FUNCTION_PERFORMANCE: {func.__name__} - {duration:.3f}s")
            
            # Warn about slow functions
            if duration > 1.0:
                perf_logger.warning(f"SLOW_FUNCTION: {func.__name__} took {duration:.3f}s")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error with performance data
            error_logger = logging.getLogger('error')
            error_logger.error(f"FUNCTION_ERROR: {func.__name__} failed after {duration:.3f}s - {e}", 
                             exc_info=True)
            raise
    
    return wrapper


def debug_route(func):
    """Decorator to add debug logging to routes"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        from flask import request, current_app
        
        current_app.logger.debug(f"ROUTE_ENTRY: {func.__name__} - {request.method} {request.url}")
        
        try:
            result = func(*args, **kwargs)
            current_app.logger.debug(f"ROUTE_SUCCESS: {func.__name__}")
            return result
            
        except Exception as e:
            current_app.logger.error(f"ROUTE_ERROR: {func.__name__} - {e}", exc_info=True)
            raise
    
    return wrapper


# Global debug logger instance
debug_logger = DebugLogger()