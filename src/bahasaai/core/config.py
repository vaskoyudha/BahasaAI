"""
Configuration management for BahasaAI.

Provides BahasaAIConfig frozen dataclass with all configuration parameters
and load_config() function to load configuration from environment variables,
YAML files, and defaults.

Priority: environment variables > YAML file > defaults

Environment variables use BAHASAAI_ prefix (e.g., BAHASAAI_DEFAULT_MODEL).
"""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from bahasaai.core.types import PipelineMode


@dataclass(frozen=True)
class BahasaAIConfig:
    """Frozen configuration for BahasaAI."""

    default_model: str = "gpt-4o"
    default_mode: PipelineMode = PipelineMode.FULL
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cache_enabled: bool = True
    cache_ttl: int = 3600
    cache_max_size: int = 1000
    debug: bool = False
    max_retries: int = 3
    retry_backoff_base: float = 1.0
    retry_max_wait: float = 30.0
    cultural_context_max_entries: int = 30
    translation_confidence_threshold: float = 0.7


def _parse_bool(value: str) -> bool:
    """Parse string value to boolean.

    Args:
        value: String value ("true", "false", "1", "0", "yes", "no")

    Returns:
        Boolean value
    """
    return value.lower() in ("true", "1", "yes")


def _parse_env_value(key: str, value: str, field_type: type) -> any:  # noqa: ANN401
    """Parse environment variable value to appropriate type.

    Args:
        key: The field name (for error messages)
        value: The string value from environment
        field_type: The target type

    Returns:
        Parsed value

    Raises:
        ValueError: If parsing fails
    """
    if field_type is bool:
        return _parse_bool(value)
    elif field_type is int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Cannot parse {key} as int: {value}") from None
    elif field_type is float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Cannot parse {key} as float: {value}") from None
    else:
        return value


def load_config(yaml_path: str | None = None) -> BahasaAIConfig:
    """Load configuration from environment variables, YAML, and defaults.

    Priority: environment variables > YAML file > defaults

    Args:
        yaml_path: Optional path to YAML config file. If None, looks for
                   bahasaai.yaml in current directory.

    Returns:
        BahasaAIConfig instance

    Raises:
        ValueError: If configuration validation fails
    """
    # Start with defaults
    config_dict = {
        "default_model": "gpt-4o",
        "default_mode": PipelineMode.FULL,
        "api_host": "0.0.0.0",
        "api_port": 8000,
        "cache_enabled": True,
        "cache_ttl": 3600,
        "cache_max_size": 1000,
        "debug": False,
        "max_retries": 3,
        "retry_backoff_base": 1.0,
        "retry_max_wait": 30.0,
        "cultural_context_max_entries": 30,
        "translation_confidence_threshold": 0.7,
    }

    # Load from YAML file if exists
    yaml_file = yaml_path or "bahasaai.yaml"
    if Path(yaml_file).exists():
        with open(yaml_file) as f:
            yaml_config = yaml.safe_load(f) or {}
            config_dict.update(yaml_config)

    # Override with environment variables
    env_prefix = "BAHASAAI_"
    field_types = {
        "default_model": str,
        "default_mode": str,
        "api_host": str,
        "api_port": int,
        "cache_enabled": bool,
        "cache_ttl": int,
        "cache_max_size": int,
        "debug": bool,
        "max_retries": int,
        "retry_backoff_base": float,
        "retry_max_wait": float,
        "cultural_context_max_entries": int,
        "translation_confidence_threshold": float,
    }

    for field, field_type in field_types.items():
        env_key = env_prefix + field.upper()
        if env_key in os.environ:
            value = os.environ[env_key]
            parsed = _parse_env_value(field, value, field_type)
            config_dict[field] = parsed

    # Handle PipelineMode enum conversion
    if isinstance(config_dict["default_mode"], str):
        try:
            config_dict["default_mode"] = PipelineMode(config_dict["default_mode"])
        except ValueError:
            raise ValueError(
                f"Invalid default_mode: {config_dict['default_mode']}. "
                f"Must be one of: {[m.value for m in PipelineMode]}"
            ) from None

    # Validate configuration
    api_port = config_dict["api_port"]
    if not (1 <= api_port <= 65535):
        raise ValueError(f"api_port must be between 1-65535, got {api_port}")

    max_retries = config_dict["max_retries"]
    if not (0 <= max_retries <= 10):
        raise ValueError(f"max_retries must be between 0-10, got {max_retries}")

    cache_ttl = config_dict["cache_ttl"]
    if not (0 <= cache_ttl <= 86400):
        raise ValueError(f"cache_ttl must be between 0-86400, got {cache_ttl}")

    threshold = config_dict["translation_confidence_threshold"]
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"translation_confidence_threshold must be between 0.0-1.0, got {threshold}"
        )

    # Clamp cultural_context_max_entries (silent clamping, no error)
    cultural_entries = config_dict["cultural_context_max_entries"]
    if cultural_entries > 30:
        config_dict["cultural_context_max_entries"] = 30

    return BahasaAIConfig(**config_dict)
