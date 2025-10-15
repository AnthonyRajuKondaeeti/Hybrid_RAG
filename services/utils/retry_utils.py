"""
Retry and Performance Utilities

This module contains retry mechanisms, performance timing, and error handling utilities.
"""

import time
import logging
from contextlib import contextmanager
from functools import wraps
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)

class RetryConfig:
    """Configuration for retry logic"""
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 exponential_base: float = 2.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.exponential_base = exponential_base
        self.max_delay = max_delay

def with_retry(retry_config: RetryConfig = None, 
               transient_errors: Tuple[str, ...] = ("no such group", "rate", "capacity", "429")):
    """Decorator for retry logic with exponential backoff"""
    if retry_config is None:
        retry_config = RetryConfig()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_config.max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    
                    # Check if error is transient
                    is_transient = any(err_type in error_msg for err_type in transient_errors)
                    
                    if is_transient and attempt < retry_config.max_retries - 1:
                        delay = min(
                            retry_config.base_delay * (retry_config.exponential_base ** attempt),
                            retry_config.max_delay
                        )
                        logger.warning(f"Transient error (attempt {attempt + 1}/{retry_config.max_retries}): {str(e)}")
                        logger.info(f"Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        # Non-transient error or final attempt
                        break
            
            # Re-raise the last exception
            raise last_exception
        return wrapper
    return decorator

@contextmanager
def performance_timer(operation_name: str, stats_dict: Dict = None):
    """Context manager for timing operations"""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info(f"{operation_name} completed in {duration:.2f}s")
        if stats_dict is not None:
            stats_dict.setdefault('timings', []).append(duration)

__all__ = ['RetryConfig', 'with_retry', 'performance_timer']