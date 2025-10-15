"""
Configuration Management for RAG Services

This module handles configuration management and provides a centralized interface
for accessing configuration values across the RAG system.
"""

from typing import Any
from ..utils.retry_utils import RetryConfig

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config):
        self.config = config
        self._cache = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with caching"""
        if key not in self._cache:
            self._cache[key] = getattr(self.config, key, default)
        return self._cache[key]
    
    def get_retry_config(self) -> RetryConfig:
        """Get retry configuration"""
        return RetryConfig(
            max_retries=self.get('MAX_RETRIES', 3),
            base_delay=self.get('RETRY_BASE_DELAY', 1.0),
            exponential_base=self.get('RETRY_EXPONENTIAL_BASE', 2.0),
            max_delay=self.get('RETRY_MAX_DELAY', 60.0)
        )

__all__ = ['ConfigManager']