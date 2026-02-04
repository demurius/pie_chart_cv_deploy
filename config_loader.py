"""
Configuration loader for the application.
Loads configuration from .env file and environment variables.
"""
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load .env file once at module import time
_env_loaded = False

def _ensure_env_loaded():
    """Ensure .env file is loaded only once."""
    global _env_loaded
    if not _env_loaded:
        if os.path.exists(".env"):
            try:
                load_dotenv(".env")
                _env_loaded = True
            except Exception as e:
                print(f"[Config] Warning: Failed to load .env file: {e}")
        _env_loaded = True


class Config:        
    """Application configuration manager."""
    
    def __init__(self, env_file: str = None):
        """
        Initialize configuration from .env file.
        
        Args:
            env_file: Path to .env file (default: .env)
        """
        self.env_file = env_file or ".env"
        # Ensure .env is loaded once at module level
        _ensure_env_loaded()
    
    def get(self, key: str, default: Any = None, required: bool = False) -> Any:
        """
        Get configuration value from environment variables.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            required: If True, raise error if key not found
            
        Returns:
            Configuration value
        """
        # Get from environment variable (uppercase with underscores)
        env_key = key.upper().replace("-", "_")
        value = os.getenv(env_key)
        
        # Use default if None
        if value is None:
            value = default
        
        # Raise error if required and still None
        if required and value is None:
            raise ValueError(f"Required configuration key '{key}' not found in environment variables")
        
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
        
    @property
    def google_credentials_json(self) -> Optional[str]:
        """Get Google credentials JSON content from environment variable."""
        return self.get("google_credentials_json")
    
    @property
    def api_token(self) -> Optional[str]:
        """Get API token for endpoint authentication."""
        return self.get("api_token")
        
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate configuration.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        if not self.user_id:
            errors.append("USER_ID is required in .env file or environment variables")
        
        # Warn about optional fields
        if not self.gemini_api_key:
            print("[Config] Warning: gemini_api_key not set - MBTI extraction will be disabled")
        
        if not self.google_sheet_id:
            print("[Config] Warning: google_sheet_id not set - Google Sheets sync will be disabled")
        
        if not self.api_token:
            print("[Config] Warning: api_token not set - API endpoints will not be protected")
        
        return len(errors) == 0, errors


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def init_config(env_file: str = None) -> Config:
    """Initialize the global configuration instance."""
    global _config_instance
    _config_instance = Config(env_file)
    return _config_instance
