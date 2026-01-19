"""
Configuration loader for the application.
Loads configuration from config.json file with fallback to environment variables.
"""
import json
import os
from typing import Dict, Any, Optional


class Config:        
    """Application configuration manager."""
    
    def __init__(self, config_file: str = None):
        """
        Initialize configuration from file.
        
        Args:
            config_file: Path to config.json file (default: CONFIG_FILE env var or ./config.json)
        """
        self.config_file = config_file or os.getenv("CONFIG_FILE", "config.json")
        self._config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from JSON file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self._config = json.load(f)
                print(f"[Config] Loaded configuration from {self.config_file}")
            except Exception as e:
                print(f"[Config] Warning: Failed to load config file: {e}")
                self._config = {}
        else:
            print(f"[Config] Warning: Config file {self.config_file} not found. Using environment variables only.")
            self._config = {}
    
    def get(self, key: str, default: Any = None, required: bool = False) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            required: If True, raise error if key not found
            
        Returns:
            Configuration value
        """
        # Try config file first
        value = self._config.get(key)
        
        # Fall back to environment variable (uppercase with underscores)
        if value is None:
            env_key = key.upper().replace("-", "_")
            value = os.getenv(env_key)
        
        # Use default if still None
        if value is None:
            value = default
        
        # Raise error if required and still None
        if required and value is None:
            raise ValueError(f"Required configuration key '{key}' not found in config file or environment")
        
        return value
    
    def get_int(self, key: str, default: int = None) -> Optional[int]:
        """Get integer configuration value."""
        value = self.get(key, default)
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            print(f"[Config] Warning: Invalid integer value for {key}: {value}")
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)
    
    @property
    def user_id(self) -> Optional[str]:
        """Get default user ID for API endpoints."""
        return self.get("user_id")
    
    @property
    def google_sheet_id(self) -> Optional[str]:
        """Get Google Sheet ID for syncing results."""
        return self.get("google_sheet_id")
    
    @property
    def gemini_api_key(self) -> Optional[str]:
        """Get Gemini API key."""
        return self.get("gemini_api_key")
    
    @property
    def subject_filter(self) -> str:
        """Get email subject filter."""
        return self.get("subject_filter", "Your recent application for")
    
    @property
    def max_workers(self) -> int:
        """Get max workers for parallel processing."""
        return self.get_int("max_workers", 4)
    
    @property
    def port(self) -> int:
        """Get port for the server."""
        return self.get_int("port", 8080)
    
    @property
    def skip_mbti(self) -> bool:
        """Get skip MBTI flag."""
        return self.get_bool("skip_mbti", False)
    
    @property
    def save_attachments(self) -> bool:
        """Get save attachments flag."""
        return self.get_bool("save_attachments", True)
    
    @property
    def auto_sync_to_sheet(self) -> bool:
        """Get auto sync to Google Sheets flag."""
        return self.get_bool("auto_sync_to_sheet", True)
    
    @property
    def debug_mode(self) -> bool:
        """Get debug mode flag (from config or DEBUG_MODE env var)."""
        return self.get_bool("debug_mode", False)
        
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate configuration.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        if not self.user_id:
            errors.append("user_id is required in config.json")
        
        # Warn about optional fields
        if not self.gemini_api_key:
            print("[Config] Warning: gemini_api_key not set - MBTI extraction will be disabled")
        
        if not self.google_sheet_id:
            print("[Config] Warning: google_sheet_id not set - Google Sheets sync will be disabled")
        
        return len(errors) == 0, errors


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def init_config(config_file: str = None) -> Config:
    """Initialize the global configuration instance."""
    global _config_instance
    _config_instance = Config(config_file)
    return _config_instance
