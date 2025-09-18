"""
Comprehensive Logging System for Horse Racing Prediction Application
Provides error tracking, audit trails, and deployment monitoring.
"""

import logging
import logging.handlers
import os
import json
import traceback
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import threading
from contextlib import contextmanager


class ComprehensiveLogger:
    """
    Advanced logging system with multiple handlers and formatters
    for different types of events and environments.
    """
    
    def __init__(self, app_name: str = "horse_racing_prediction"):
        self.app_name = app_name
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different log types
        (self.log_dir / "errors").mkdir(exist_ok=True)
        (self.log_dir / "audit").mkdir(exist_ok=True)
        (self.log_dir / "deployment").mkdir(exist_ok=True)
        (self.log_dir / "performance").mkdir(exist_ok=True)
        (self.log_dir / "security").mkdir(exist_ok=True)
        
        self.loggers = {}
        self._setup_loggers()
        
    def _setup_loggers(self):
        """Set up different loggers for different purposes"""
        
        # Main application logger
        self.loggers['app'] = self._create_logger(
            'app',
            self.log_dir / 'application.log',
            logging.INFO
        )
        
        # Error logger with detailed stack traces
        self.loggers['error'] = self._create_logger(
            'error',
            self.log_dir / 'errors' / 'errors.log',
            logging.ERROR,
            include_stack_trace=True
        )
        
        # Audit logger for tracking user actions and system changes
        self.loggers['audit'] = self._create_logger(
            'audit',
            self.log_dir / 'audit' / 'audit.log',
            logging.INFO,
            formatter_type='audit'
        )
        
        # Deployment logger for tracking deployment processes
        self.loggers['deployment'] = self._create_logger(
            'deployment',
            self.log_dir / 'deployment' / 'deployment.log',
            logging.INFO,
            formatter_type='deployment'
        )
        
        # Performance logger for monitoring system performance
        self.loggers['performance'] = self._create_logger(
            'performance',
            self.log_dir / 'performance' / 'performance.log',
            logging.INFO,
            formatter_type='performance'
        )
        
        # Security logger for security-related events
        self.loggers['security'] = self._create_logger(
            'security',
            self.log_dir / 'security' / 'security.log',
            logging.WARNING,
            formatter_type='security'
        )
        
    def _create_logger(self, name: str, log_file: Path, level: int, 
                      include_stack_trace: bool = False, 
                      formatter_type: str = 'standard') -> logging.Logger:
        """Create a logger with file and console handlers"""
        
        logger = logging.getLogger(f"{self.app_name}.{name}")
        logger.setLevel(level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        
        # Console handler for errors and critical messages
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING if name != 'deployment' else logging.INFO)
        
        # Set formatters
        file_formatter = self._get_formatter(formatter_type, include_stack_trace)
        console_formatter = self._get_formatter('console', include_stack_trace)
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _get_formatter(self, formatter_type: str, include_stack_trace: bool = False) -> logging.Formatter:
        """Get appropriate formatter based on type"""
        
        formatters = {
            'standard': logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ),
            'audit': logging.Formatter(
                '%(asctime)s | AUDIT | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ),
            'deployment': logging.Formatter(
                '%(asctime)s | DEPLOY | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ),
            'performance': logging.Formatter(
                '%(asctime)s | PERF | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ),
            'security': logging.Formatter(
                '%(asctime)s | SECURITY | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ),
            'console': logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            )
        }
        
        return formatters.get(formatter_type, formatters['standard'])
    
    def log_error(self, message: str, exception: Optional[Exception] = None, 
                  context: Optional[Dict[str, Any]] = None):
        """Log an error with full context and stack trace"""
        
        error_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': message,
            'context': context or {}
        }
        
        if exception:
            error_data.update({
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'stack_trace': traceback.format_exc()
            })
        
        # Log to error logger
        self.loggers['error'].error(json.dumps(error_data, indent=2))
        
        # Also log to main app logger
        self.loggers['app'].error(f"ERROR: {message}")
        
    def log_audit(self, action: str, user_id: Optional[str] = None, 
                  resource: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Log an audit event"""
        
        audit_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': action,
            'user_id': user_id,
            'resource': resource,
            'details': details or {},
            'thread_id': threading.get_ident()
        }
        
        self.loggers['audit'].info(json.dumps(audit_data))
        
    def log_deployment(self, stage: str, status: str, message: str, 
                      details: Optional[Dict[str, Any]] = None):
        """Log deployment events"""
        
        deployment_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'stage': stage,
            'status': status,
            'message': message,
            'details': details or {}
        }
        
        level = logging.INFO if status == 'SUCCESS' else logging.ERROR
        self.loggers['deployment'].log(level, json.dumps(deployment_data))
        
    def log_performance(self, operation: str, duration: float, 
                       details: Optional[Dict[str, Any]] = None):
        """Log performance metrics"""
        
        perf_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'operation': operation,
            'duration_seconds': duration,
            'details': details or {}
        }
        
        self.loggers['performance'].info(json.dumps(perf_data))
        
    def log_security(self, event: str, severity: str, details: Optional[Dict[str, Any]] = None):
        """Log security events"""
        
        security_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event': event,
            'severity': severity,
            'details': details or {}
        }
        
        level = getattr(logging, severity.upper(), logging.WARNING)
        self.loggers['security'].log(level, json.dumps(security_data))
        
    @contextmanager
    def performance_context(self, operation: str):
        """Context manager for performance monitoring"""
        start_time = datetime.now()
        try:
            yield
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            self.log_performance(operation, duration)
            
    def info(self, message: str):
        """Log info message to main logger"""
        self.loggers['app'].info(message)
        
    def warning(self, message: str):
        """Log warning message to main logger"""
        self.loggers['app'].warning(message)
        
    def debug(self, message: str):
        """Log debug message to main logger"""
        self.loggers['app'].debug(message)


# Global logger instance
comprehensive_logger = ComprehensiveLogger()


# Decorator for automatic error logging
def log_errors(logger_instance=None):
    """Decorator to automatically log errors from functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = logger_instance or comprehensive_logger
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log_error(
                    f"Error in function {func.__name__}",
                    exception=e,
                    context={
                        'function': func.__name__,
                        'args': str(args)[:200],  # Limit length
                        'kwargs': str(kwargs)[:200]
                    }
                )
                raise
        return wrapper
    return decorator


# Decorator for performance monitoring
def monitor_performance(operation_name=None):
    """Decorator to monitor function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            with comprehensive_logger.performance_context(op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator