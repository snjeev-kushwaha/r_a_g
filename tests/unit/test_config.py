"""
Unit tests for the strict configuration module.
Verifies that all settings are properly read from environment variables (.env),
and that missing or malformed variables raise explicit ValueError exceptions.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

import config


class TestConfigModule:
    """Test suite for config.py strict loading and validation."""

    def test_current_config_loaded_successfully(self):
        """Verify that currently configured settings are correctly populated from .env."""
        assert config.API_HOST is not None
        assert isinstance(config.API_PORT, int)
        assert config.OLLAMA_BASE_URL.startswith("http")
        assert config.OLLAMA_LLM_MODEL != ""
        assert config.OLLAMA_EMBED_MODEL != ""
        assert isinstance(config.OLLAMA_TIMEOUT, float)
        assert config.OLLAMA_MODEL == config.OLLAMA_LLM_MODEL
        assert isinstance(config.CHUNK_SIZE, int)
        assert isinstance(config.CHUNK_OVERLAP, int)
        assert isinstance(config.TOP_K, int)
        assert isinstance(config.MAX_EMBED_CHARS, int)
        assert isinstance(config.MAX_CONTEXT_CHARS, int)
        assert isinstance(config.LLM_NUM_PREDICT, int)
        assert isinstance(config.LLM_TEMPERATURE, float)
        assert isinstance(config.VECTOR_DIM, int)
        assert Path(config.UPLOAD_DIR).is_absolute()
        assert Path(config.VECTOR_DB_PATH).is_absolute()
        assert Path(config.LOG_FILE).is_absolute()

    def test_get_required_env_raises_when_missing(self):
        """Verify that _get_required_env raises ValueError if variable is missing or empty."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing required environment variable 'NONEXISTENT_VAR'"):
                config._get_required_env("NONEXISTENT_VAR")

    def test_get_required_env_raises_when_blank(self):
        """Verify that _get_required_env raises ValueError if variable contains only whitespace."""
        with patch.dict(os.environ, {"BLANK_VAR": "   "}):
            with pytest.raises(ValueError, match="Missing required environment variable 'BLANK_VAR'"):
                config._get_required_env("BLANK_VAR")

    def test_get_required_int_validation(self):
        """Verify that _get_required_int parses valid integers and raises on invalid ones."""
        with patch.dict(os.environ, {"TEST_INT": "123"}):
            assert config._get_required_int("TEST_INT") == 123

        with patch.dict(os.environ, {"TEST_INVALID_INT": "not-an-int"}):
            with pytest.raises(ValueError, match="must be an integer"):
                config._get_required_int("TEST_INVALID_INT")

    def test_get_required_float_validation(self):
        """Verify that _get_required_float parses valid floats and raises on invalid ones."""
        with patch.dict(os.environ, {"TEST_FLOAT": "42.5"}):
            assert config._get_required_float("TEST_FLOAT") == 42.5

        with patch.dict(os.environ, {"TEST_INVALID_FLOAT": "abc"}):
            with pytest.raises(ValueError, match="must be a float"):
                config._get_required_float("TEST_INVALID_FLOAT")

    def test_resolve_env_path_relative(self):
        """Verify that relative paths are resolved against BASE_DIR."""
        with patch.dict(os.environ, {"REL_PATH": "my_folder/sub"}):
            resolved = config._resolve_env_path("REL_PATH")
            assert Path(resolved).is_absolute()
            assert resolved.endswith(str(Path("my_folder/sub")))

    def test_resolve_env_path_absolute(self):
        """Verify that absolute paths remain unchanged."""
        abs_path = str(Path("C:/test/abs/path").resolve())
        with patch.dict(os.environ, {"ABS_PATH": abs_path}):
            resolved = config._resolve_env_path("ABS_PATH")
            assert Path(resolved).is_absolute()
            assert resolved == abs_path
