import json
import logging
import logging.config
import os
import sys
import time
import traceback
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

class ContextualLogger:
    """Enterprise-grade contextual logger with distributed tracing capabilities"""
    
    def __init__(self, name: str, context: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(name)
        self.context = context or {}
        self.trace_id = self.context.get('trace_id') or str(uuid.uuid4())
        self.context['trace_id'] = self.trace_id
        
    def _format_message(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format log message with context and extras"""
        timestamp = datetime.utcnow().isoformat() + 'Z'
        
        log_entry = {
            'timestamp': timestamp,
            'message': msg,
            'trace_id': self.trace_id,
            **self.context
        }
        
        if extra:
            # Don't overwrite existing context keys
            for k, v in extra.items():
                if k not in log_entry:
                    log_entry[k] = v
                    
        return log_entry
    
    def debug(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message with context"""
        self.logger.debug(json.dumps(self._format_message(msg, extra)))
        
    def info(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message with context"""
        self.logger.info(json.dumps(self._format_message(msg, extra)))
        
    def warning(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message with context"""
        self.logger.warning(json.dumps(self._format_message(msg, extra)))
        
    def error(self, msg: str, error: Optional[Exception] = None, extra: Optional[Dict[str, Any]] = None):
        """Log error message with exception details and context"""
        log_extra = extra or {}
        
        if error:
            log_extra.update({
                'error_type': error.__class__.__name__,
                'error_message': str(error),
                'stack_trace': traceback.format_exc()
            })
            
        self.logger.error(json.dumps(self._format_message(msg, log_extra)))
        
    def critical(self, msg: str, error: Optional[Exception] = None, extra: Optional[Dict[str, Any]] = None):
        """Log critical message with exception details and context"""
        log_extra = extra or {}
        
        if error:
            log_extra.update({
                'error_type': error.__class__.__name__,
                'error_message': str(error),
                'stack_trace': traceback.format_exc()
            })
            
        self.logger.critical(json.dumps(self._format_message(msg, log_extra)))
        
    def with_context(self, **kwargs) -> 'ContextualLogger':
        """Create a new logger with additional context"""
        new_context = {**self.context, **kwargs}
        return ContextualLogger(self.logger.name, new_context)
    
    def time_execution(self, func=None, *, task_name=None):
        """Decorator to time function execution and log results"""
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                task = task_name or f.__name__
                start_time = time.time()
                self.info(f"Starting {task}", {"task": task, "action": "start"})
                
                try:
                    result = f(*args, **kwargs)
                    execution_time = time.time() - start_time
                    self.info(
                        f"Completed {task} in {execution_time:.2f}s",
                        {"task": task, "action": "complete", "duration_seconds": execution_time}
                    )
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.error(
                        f"Failed {task} after {execution_time:.2f}s",
                        error=e,
                        extra={"task": task, "action": "error", "duration_seconds": execution_time}
                    )
                    raise
            return wrapper
            
        if func:
            return decorator(func)
        return decorator


class LogManager:
    """Manager for configuring and accessing loggers"""
    
    _instance = None
    _loggers: Dict[str, ContextualLogger] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance._configure()
        return cls._instance
    
    def _configure(self):
        """Configure logging based on environment"""
        env = os.environ.get('ENVIRONMENT', 'development').lower()
        
        # Ensure log directory exists regardless of environment
        project_root = Path(__file__).parent.parent.parent  # /emerge/src/utils -> /emerge
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "emerge.log"
        
        # Base configuration
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'simple': {
                    'format': '%(message)s'
                },
                'verbose': {
                    'format': '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                    'datefmt': '%Y-%m-%dT%H:%M:%S%z'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'DEBUG',
                    'formatter': 'simple',
                    'stream': 'ext://sys.stdout'
                }
            },
            'loggers': {
                '': {  # Root logger
                    'handlers': ['console'],
                    'level': 'INFO',
                }
            }
        }
        
        # Add file handler for both development and production
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO' if env == 'production' else 'DEBUG',
            'formatter': 'verbose',
            'filename': str(log_file),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
        }
        config['loggers']['']['handlers'].append('file')
        
        # Environment-specific configuration
        if env == 'production':
            # Add cloud logging handler for production
            try:
                config['handlers']['cloud'] = {
                    'class': 'google.cloud.logging.handlers.CloudLoggingHandler',
                    'formatter': 'simple',
                    'level': 'INFO'
                }
                config['loggers']['']['handlers'].append('cloud')
            except ImportError:
                print("Google Cloud Logging not available. Skipping cloud handler.")
            
        elif env == 'development':
            config['loggers']['']['level'] = 'DEBUG'
            
        # Apply configuration
        try:
            logging.config.dictConfig(config)
        except Exception as e:
            # Fallback to basic configuration if the advanced one fails
            print(f"Error configuring logging: {e}")
            print("Falling back to basic configuration")
            logging.basicConfig(
                level=logging.DEBUG,
                format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                handlers=[logging.StreamHandler(sys.stdout)]
            )
    
    def get_logger(self, name: str, context: Optional[Dict[str, Any]] = None) -> ContextualLogger:
        """Get or create a logger with the given name and context"""
        if name not in self._loggers:
            self._loggers[name] = ContextualLogger(name, context)
        return self._loggers[name] if not context else self._loggers[name].with_context(**context)


# Convenience function for getting loggers
def get_logger(name: str = None, context: Optional[Dict[str, Any]] = None) -> ContextualLogger:
    """Get a contextual logger with the specified name and context"""
    if name is None:
        # Attempt to infer the caller module name
        frame = sys._getframe(1)
        name = frame.f_globals.get('__name__', 'root')
    return LogManager().get_logger(name, context)