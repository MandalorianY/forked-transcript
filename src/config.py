"""
Configuration management with three-tier precedence system:
1. Default values from codebase
2. Environment variables from .env
3. User settings from UI (stored in session state for Streamlit)

Precedence: UI Settings > Environment Variables > Defaults
"""

import os
from typing import Any, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ConfigManager:
    """Manages configuration with three-tier precedence."""

    # Default values (Tier 1 - Codebase defaults)
    DEFAULTS = {
        "API_BASE_URL": "http://localhost:5001",
        "LLM_API_BASE_URL": "https://api.openai.com/v1",
        "LLM_MODEL": "gpt-4",
        "OPENAI_API_KEY": "",
        "HUGGINGFACE_TOKEN": "",
        # Model configurations
        "WHISPER_MODEL": "base",  # Options: tiny, base, small, medium, large, or HuggingFace model ID
        "DIARIZATION_MODEL": "pyannote/speaker-diarization-3.1",  # HuggingFace model ID or local path
    }

    @staticmethod
    def get(key: str, ui_override: Optional[Any] = None) -> Any:
        """
        Get configuration value with three-tier precedence.

        Args:
            key: Configuration key
            ui_override: UI setting value (highest priority)

        Returns:
            Configuration value from highest priority source

        Priority:
            1. UI override (if provided and not empty)
            2. Environment variable
            3. Default value
        """
        # Tier 3: UI override (highest priority)
        if ui_override is not None and ui_override != "":
            return ui_override

        # Tier 2: Environment variable
        env_value = os.getenv(key)
        if env_value is not None and env_value != "":
            return env_value

        # Tier 1: Default value
        return ConfigManager.DEFAULTS.get(key, "")

    @staticmethod
    def get_display_value(key: str, ui_override: Optional[Any] = None) -> tuple[Any, str]:
        """
        Get configuration value and its source.

        Returns:
            Tuple of (value, source) where source is 'ui', 'env', or 'default'
        """
        # Check UI override
        if ui_override is not None and ui_override != "":
            return ui_override, "ui"

        # Check environment variable
        env_value = os.getenv(key)
        if env_value is not None and env_value != "":
            return env_value, "env"

        # Default value
        default_value = ConfigManager.DEFAULTS.get(key, "")
        return default_value, "default"

    @staticmethod
    def is_using_default(key: str, ui_override: Optional[Any] = None) -> bool:
        """Check if configuration is using default value."""
        _, source = ConfigManager.get_display_value(key, ui_override)
        return source == "default"

    @staticmethod
    def is_using_env(key: str, ui_override: Optional[Any] = None) -> bool:
        """Check if configuration is using environment variable."""
        _, source = ConfigManager.get_display_value(key, ui_override)
        return source == "env"

    @staticmethod
    def is_using_ui(key: str, ui_override: Optional[Any] = None) -> bool:
        """Check if configuration is using UI override."""
        _, source = ConfigManager.get_display_value(key, ui_override)
        return source == "ui"
