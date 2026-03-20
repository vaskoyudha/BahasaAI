"""
Tests for configuration management module.

Tests cover:
- Default values
- Environment variable overrides
- Type parsing (int, float, bool)
- Validation (port range, retry limits, etc.)
- Frozen dataclass behavior
- YAML file loading with env var override priority
- Silent clamping of cultural_context_max_entries
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from bahasaai.core.config import BahasaAIConfig, load_config
from bahasaai.core.types import PipelineMode


class TestBahasaAIConfigDefaults:
    """Test default configuration values."""

    def test_default_model(self):
        """Default model should be gpt-4o."""
        cfg = BahasaAIConfig()
        assert cfg.default_model == "gpt-4o"

    def test_default_mode(self):
        """Default mode should be FULL."""
        cfg = BahasaAIConfig()
        assert cfg.default_mode == PipelineMode.FULL

    def test_default_api_host(self):
        """Default API host should be 0.0.0.0."""
        cfg = BahasaAIConfig()
        assert cfg.api_host == "0.0.0.0"

    def test_default_api_port(self):
        """Default API port should be 8000."""
        cfg = BahasaAIConfig()
        assert cfg.api_port == 8000

    def test_default_cache_enabled(self):
        """Cache should be enabled by default."""
        cfg = BahasaAIConfig()
        assert cfg.cache_enabled is True

    def test_default_cache_ttl(self):
        """Default cache TTL should be 3600 seconds."""
        cfg = BahasaAIConfig()
        assert cfg.cache_ttl == 3600

    def test_default_cache_max_size(self):
        """Default cache max size should be 1000."""
        cfg = BahasaAIConfig()
        assert cfg.cache_max_size == 1000

    def test_default_debug(self):
        """Debug should be False by default."""
        cfg = BahasaAIConfig()
        assert cfg.debug is False

    def test_default_max_retries(self):
        """Default max retries should be 3."""
        cfg = BahasaAIConfig()
        assert cfg.max_retries == 3

    def test_default_retry_backoff_base(self):
        """Default retry backoff base should be 1.0."""
        cfg = BahasaAIConfig()
        assert cfg.retry_backoff_base == 1.0

    def test_default_retry_max_wait(self):
        """Default retry max wait should be 30.0."""
        cfg = BahasaAIConfig()
        assert cfg.retry_max_wait == 30.0

    def test_default_cultural_context_max_entries(self):
        """Default cultural context max entries should be 30."""
        cfg = BahasaAIConfig()
        assert cfg.cultural_context_max_entries == 30

    def test_default_translation_confidence_threshold(self):
        """Default translation confidence threshold should be 0.7."""
        cfg = BahasaAIConfig()
        assert cfg.translation_confidence_threshold == 0.7


class TestLoadConfigEnvironmentVariables:
    """Test load_config with environment variable overrides."""

    def test_env_override_default_model(self):
        """Environment variable BAHASAAI_DEFAULT_MODEL should override default."""
        os.environ["BAHASAAI_DEFAULT_MODEL"] = "claude-3.5-sonnet"
        try:
            cfg = load_config()
            assert cfg.default_model == "claude-3.5-sonnet"
        finally:
            del os.environ["BAHASAAI_DEFAULT_MODEL"]

    def test_env_override_api_port_as_int(self):
        """Environment variable BAHASAAI_API_PORT should be parsed as int."""
        os.environ["BAHASAAI_API_PORT"] = "9000"
        try:
            cfg = load_config()
            assert cfg.api_port == 9000
            assert isinstance(cfg.api_port, int)
        finally:
            del os.environ["BAHASAAI_API_PORT"]

    def test_env_override_debug_as_bool_true(self):
        """Environment variable BAHASAAI_DEBUG=true should parse to True."""
        os.environ["BAHASAAI_DEBUG"] = "true"
        try:
            cfg = load_config()
            assert cfg.debug is True
        finally:
            del os.environ["BAHASAAI_DEBUG"]

    def test_env_override_debug_as_bool_false(self):
        """Environment variable BAHASAAI_DEBUG=false should parse to False."""
        os.environ["BAHASAAI_DEBUG"] = "false"
        try:
            cfg = load_config()
            assert cfg.debug is False
        finally:
            del os.environ["BAHASAAI_DEBUG"]

    def test_env_override_debug_as_bool_1(self):
        """Environment variable BAHASAAI_DEBUG=1 should parse to True."""
        os.environ["BAHASAAI_DEBUG"] = "1"
        try:
            cfg = load_config()
            assert cfg.debug is True
        finally:
            del os.environ["BAHASAAI_DEBUG"]

    def test_env_override_debug_as_bool_0(self):
        """Environment variable BAHASAAI_DEBUG=0 should parse to False."""
        os.environ["BAHASAAI_DEBUG"] = "0"
        try:
            cfg = load_config()
            assert cfg.debug is False
        finally:
            del os.environ["BAHASAAI_DEBUG"]

    def test_env_override_cache_ttl(self):
        """Environment variable BAHASAAI_CACHE_TTL should override default."""
        os.environ["BAHASAAI_CACHE_TTL"] = "7200"
        try:
            cfg = load_config()
            assert cfg.cache_ttl == 7200
        finally:
            del os.environ["BAHASAAI_CACHE_TTL"]

    def test_env_override_translation_confidence_threshold(self):
        """Environment variable BAHASAAI_TRANSLATION_CONFIDENCE_THRESHOLD should be parsed as float."""
        os.environ["BAHASAAI_TRANSLATION_CONFIDENCE_THRESHOLD"] = "0.85"
        try:
            cfg = load_config()
            assert cfg.translation_confidence_threshold == 0.85
            assert isinstance(cfg.translation_confidence_threshold, float)
        finally:
            del os.environ["BAHASAAI_TRANSLATION_CONFIDENCE_THRESHOLD"]


class TestLoadConfigValidation:
    """Test load_config validation rules."""

    def test_invalid_port_too_high(self):
        """Port above 65535 should raise ValueError."""
        os.environ["BAHASAAI_API_PORT"] = "99999"
        try:
            with pytest.raises(ValueError, match="api_port"):
                load_config()
        finally:
            del os.environ["BAHASAAI_API_PORT"]

    def test_invalid_port_zero(self):
        """Port 0 should raise ValueError."""
        os.environ["BAHASAAI_API_PORT"] = "0"
        try:
            with pytest.raises(ValueError, match="api_port"):
                load_config()
        finally:
            del os.environ["BAHASAAI_API_PORT"]

    def test_invalid_port_negative(self):
        """Negative port should raise ValueError."""
        os.environ["BAHASAAI_API_PORT"] = "-1"
        try:
            with pytest.raises(ValueError, match="api_port"):
                load_config()
        finally:
            del os.environ["BAHASAAI_API_PORT"]

    def test_invalid_max_retries_too_high(self):
        """max_retries > 10 should raise ValueError."""
        os.environ["BAHASAAI_MAX_RETRIES"] = "15"
        try:
            with pytest.raises(ValueError, match="max_retries"):
                load_config()
        finally:
            del os.environ["BAHASAAI_MAX_RETRIES"]

    def test_invalid_max_retries_negative(self):
        """Negative max_retries should raise ValueError."""
        os.environ["BAHASAAI_MAX_RETRIES"] = "-1"
        try:
            with pytest.raises(ValueError, match="max_retries"):
                load_config()
        finally:
            del os.environ["BAHASAAI_MAX_RETRIES"]

    def test_invalid_cache_ttl_too_high(self):
        """cache_ttl > 86400 should raise ValueError."""
        os.environ["BAHASAAI_CACHE_TTL"] = "100000"
        try:
            with pytest.raises(ValueError, match="cache_ttl"):
                load_config()
        finally:
            del os.environ["BAHASAAI_CACHE_TTL"]

    def test_invalid_cache_ttl_negative(self):
        """Negative cache_ttl should raise ValueError."""
        os.environ["BAHASAAI_CACHE_TTL"] = "-1"
        try:
            with pytest.raises(ValueError, match="cache_ttl"):
                load_config()
        finally:
            del os.environ["BAHASAAI_CACHE_TTL"]

    def test_invalid_translation_confidence_threshold_above_1(self):
        """translation_confidence_threshold > 1.0 should raise ValueError."""
        os.environ["BAHASAAI_TRANSLATION_CONFIDENCE_THRESHOLD"] = "1.5"
        try:
            with pytest.raises(ValueError, match="translation_confidence_threshold"):
                load_config()
        finally:
            del os.environ["BAHASAAI_TRANSLATION_CONFIDENCE_THRESHOLD"]

    def test_invalid_translation_confidence_threshold_below_0(self):
        """translation_confidence_threshold < 0.0 should raise ValueError."""
        os.environ["BAHASAAI_TRANSLATION_CONFIDENCE_THRESHOLD"] = "-0.1"
        try:
            with pytest.raises(ValueError, match="translation_confidence_threshold"):
                load_config()
        finally:
            del os.environ["BAHASAAI_TRANSLATION_CONFIDENCE_THRESHOLD"]

    def test_valid_port_boundary_1(self):
        """Port 1 should be valid."""
        os.environ["BAHASAAI_API_PORT"] = "1"
        try:
            cfg = load_config()
            assert cfg.api_port == 1
        finally:
            del os.environ["BAHASAAI_API_PORT"]

    def test_valid_port_boundary_65535(self):
        """Port 65535 should be valid."""
        os.environ["BAHASAAI_API_PORT"] = "65535"
        try:
            cfg = load_config()
            assert cfg.api_port == 65535
        finally:
            del os.environ["BAHASAAI_API_PORT"]

    def test_valid_max_retries_boundary_0(self):
        """max_retries 0 should be valid."""
        os.environ["BAHASAAI_MAX_RETRIES"] = "0"
        try:
            cfg = load_config()
            assert cfg.max_retries == 0
        finally:
            del os.environ["BAHASAAI_MAX_RETRIES"]

    def test_valid_max_retries_boundary_10(self):
        """max_retries 10 should be valid."""
        os.environ["BAHASAAI_MAX_RETRIES"] = "10"
        try:
            cfg = load_config()
            assert cfg.max_retries == 10
        finally:
            del os.environ["BAHASAAI_MAX_RETRIES"]


class TestLoadConfigClamping:
    """Test silent clamping behavior."""

    def test_cultural_context_max_entries_clamped_high(self):
        """cultural_context_max_entries > 30 should be silently clamped to 30."""
        os.environ["BAHASAAI_CULTURAL_CONTEXT_MAX_ENTRIES"] = "50"
        try:
            cfg = load_config()
            assert cfg.cultural_context_max_entries == 30
        finally:
            del os.environ["BAHASAAI_CULTURAL_CONTEXT_MAX_ENTRIES"]

    def test_cultural_context_max_entries_not_clamped_below(self):
        """cultural_context_max_entries <= 30 should not be clamped."""
        os.environ["BAHASAAI_CULTURAL_CONTEXT_MAX_ENTRIES"] = "20"
        try:
            cfg = load_config()
            assert cfg.cultural_context_max_entries == 20
        finally:
            del os.environ["BAHASAAI_CULTURAL_CONTEXT_MAX_ENTRIES"]


class TestBahasaAIConfigFrozen:
    """Test frozen dataclass behavior."""

    def test_config_is_frozen(self):
        """Config object should be frozen and raise FrozenInstanceError on assignment."""
        from dataclasses import FrozenInstanceError

        cfg = BahasaAIConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.debug = True

    def test_config_is_frozen_model(self):
        """Config model field should be frozen."""
        from dataclasses import FrozenInstanceError

        cfg = BahasaAIConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.default_model = "gpt-3.5-turbo"


class TestLoadConfigYAML:
    """Test YAML file loading and environment variable priority."""

    def test_load_config_from_yaml_file(self):
        """load_config should load from bahasaai.yaml if it exists."""
        yaml_content = {
            "default_model": "claude-3-opus",
            "api_port": 5000,
            "debug": True,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "bahasaai.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(yaml_content, f)

            # Change to temp directory and load config
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                cfg = load_config()
                assert cfg.default_model == "claude-3-opus"
                assert cfg.api_port == 5000
                assert cfg.debug is True
            finally:
                os.chdir(old_cwd)

    def test_env_vars_override_yaml(self):
        """Environment variables should override YAML values."""
        yaml_content = {
            "default_model": "claude-3-opus",
            "api_port": 5000,
        }

        os.environ["BAHASAAI_DEFAULT_MODEL"] = "gpt-4-turbo"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                yaml_path = Path(tmpdir) / "bahasaai.yaml"
                with open(yaml_path, "w") as f:
                    yaml.dump(yaml_content, f)

                old_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)
                    cfg = load_config()
                    # Env var should override YAML
                    assert cfg.default_model == "gpt-4-turbo"
                    # YAML value should still apply for non-env fields
                    assert cfg.api_port == 5000
                finally:
                    os.chdir(old_cwd)
        finally:
            del os.environ["BAHASAAI_DEFAULT_MODEL"]

    def test_load_config_explicit_yaml_path(self):
        """load_config should accept explicit yaml_path parameter."""
        yaml_content = {
            "default_model": "claude-3-sonnet",
            "cache_ttl": 1800,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "custom_config.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(yaml_content, f)

            cfg = load_config(yaml_path=str(yaml_path))
            assert cfg.default_model == "claude-3-sonnet"
            assert cfg.cache_ttl == 1800

    def test_load_config_no_yaml_uses_defaults(self):
        """load_config should use defaults when no YAML file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                cfg = load_config()
                # All defaults should apply
                assert cfg.default_model == "gpt-4o"
                assert cfg.api_port == 8000
                assert cfg.debug is False
            finally:
                os.chdir(old_cwd)
