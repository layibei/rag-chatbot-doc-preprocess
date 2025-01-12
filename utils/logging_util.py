# logging_util.py
import os
import sys
from contextvars import ContextVar, copy_context
from typing import Dict, Any, Optional

from loguru import logger
from utils.logger_init import logger  # Import pre-configured logger

# Context management
_request_context: ContextVar[Dict[str, Any]] = ContextVar('fastapi_request_context', default={})
# Get the absolute path of the current file
CURRENT_FILE_PATH = os.path.abspath(__file__)
# Get the directory containing the current file
BASE_DIR = os.path.dirname(CURRENT_FILE_PATH)

def set_context(**kwargs):
    """
    Set context values using keyword arguments
    Thread-safe context setting that creates a new context dictionary for each request
    """
    try:
        # Create a new context dictionary instead of updating existing one
        new_context = {}
        # Optionally merge existing context if needed
        # new_context.update(get_context())
        new_context.update(kwargs)
        _request_context.set(new_context)
        logger.debug(f"Context set: {new_context}")
    except Exception as e:
        logger.error(f"Error setting context: {str(e)}")

def get_context() -> Dict[str, Any]:
    """
    Get the current context dictionary
    Returns a copy to prevent modification of the context
    """
    try:
        return dict(_request_context.get())  # Return a copy
    except Exception as e:
        logger.error(f"Error getting context: {str(e)}")
        return {}

def clear_context():
    """lear all context values"""
    try:
       _request_context.set({})
       logger.debug("Context cleared")  # Debug log
    except Exception as e:
       logger.error(f"Error clearing context: {str(e)}")


class SafeContextFilter:
    def __call__(self, record):
        try:
            # Get current context and update record's extra dict
            context = get_context()
            record["extra"].update(context)
            return True
        except Exception as e:
            logger.error(f"Error in context filter: {str(e)}")
            return True

def configure_logger(log_file="app.log", max_bytes=10 * 1024 * 1024, backup_count=5):
    """Configure logger with context support"""
    # Lazy import to avoid circular dependency
    from config.common_settings import CommonConfig
    config = CommonConfig()
    
    logger.remove()

    # Format string with extra context
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "{level.icon} {level.name:<8} | "
        "<blue>{thread.name}</blue> | "
        # "<blue>{process.id}</blue> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "{extra} | "
        "{message}"
    )

    # Get logging configuration
    logging_levels = config.config.get("app", {}).get("logging.level", {})
    root_level = logging_levels.get("root", "INFO")

    def log_filter(record):
        """Filter log records based on module name with hierarchical path support"""
        module_name = record["name"]
        
        # Find the most specific matching package path
        matching_level = root_level
        matching_length = 0
        
        for pkg_path, level in logging_levels.items():
            if pkg_path != "root" and module_name.startswith(pkg_path):
                path_length = len(pkg_path.split('.'))
                if path_length > matching_length:
                    matching_level = level
                    matching_length = path_length
        
        return record["level"].name >= matching_level

    # Add file handler
    logger.add(
        sink=log_file,
        format=log_format,
        filter=lambda record: SafeContextFilter()(record) and log_filter(record),
        colorize=False,
        enqueue=True,
        rotation=max_bytes,
        retention=backup_count,
        catch=True,
        level="DEBUG"  # Base level - actual filtering done by log_filter
    )

    # Console handler
    logger.add(
        sink=sys.stdout,
        format=log_format,
        filter=lambda record: SafeContextFilter()(record) and log_filter(record),
        colorize=True,
        enqueue=True,
        catch=True,
        level="DEBUG"  # Base level - actual filtering done by log_filter
    )

    return logger


# Initialize the logger
logger = configure_logger(os.path.join(BASE_DIR, "../app.log"))
logger.level("INFO")

if __name__ == "__main__":
    logger.info("This is an info message.")
