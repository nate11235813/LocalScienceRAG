"""Logging configuration utilities."""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def setup_logging(
    config: Optional[Dict[str, Any]] = None,
    log_file: Optional[Path] = None,
    verbose: bool = False
) -> None:
    """Set up logging configuration.
    
    Args:
        config: Optional configuration dictionary
        log_file: Optional log file path
        verbose: Enable verbose logging
    """
    # Default configuration
    if config is None:
        config = {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    logging_config = config.get("logging", {})
    
    # Determine log level
    if verbose:
        level = logging.DEBUG
    else:
        level_str = logging_config.get("level", "INFO")
        level = getattr(logging, level_str, logging.INFO)
    
    # Get format
    log_format = logging_config.get(
        "format", 
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Configure handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    # Suppress some noisy loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {logging.getLevelName(level)}")
    if log_file:
        logger.info(f"Logging to file: {log_file}")


class SpinnerLogHandler(logging.Handler):
    """Custom log handler that works with CLI spinners."""
    
    def __init__(self, spinner_active: bool = False):
        super().__init__()
        self.spinner_active = spinner_active
        self.buffer = []
    
    def emit(self, record):
        """Emit a log record."""
        msg = self.format(record)
        
        if self.spinner_active:
            # Buffer messages while spinner is active
            self.buffer.append(msg)
        else:
            # Print immediately
            print(msg)
    
    def flush_buffer(self):
        """Flush buffered messages."""
        for msg in self.buffer:
            print(msg)
        self.buffer.clear()
    
    def set_spinner_active(self, active: bool):
        """Set spinner active state."""
        if not active and self.spinner_active:
            # Spinner stopping, flush buffer
            self.flush_buffer()
        self.spinner_active = active