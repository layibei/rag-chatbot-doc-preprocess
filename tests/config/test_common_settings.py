import pytest
from unittest.mock import patch
from config.common_settings import CommonConfig
import os

SAMPLE_CONFIG = """
app:
  models:
    llm:
      type: ollama
      model: qwen2.5
      temperature: 0.7
      max_tokens: 2000
    embedding:
      type: huggingface
      model: sentence-transformers/all-mpnet-base-v2
"""

@pytest.fixture
def mock_logger(monkeypatch):
    """Mock logger to avoid metadata formatting error"""
    class MockLogger:
        def info(self, msg, *args, **kwargs):
            pass
        def error(self, msg, *args, **kwargs):
            pass
        def debug(self, msg, *args, **kwargs):
            pass
    
    monkeypatch.setattr('config.common_settings.logger', MockLogger())

def test_config_initialization_with_valid_file(tmp_path, monkeypatch, mock_logger):
    # Create config file
    config_file = tmp_path / "app.yaml"
    config_file.write_text(SAMPLE_CONFIG)
    
    # Patch BASE_DIR
    monkeypatch.setattr('config.common_settings.BASE_DIR', str(tmp_path))
    
    # Use leading slash to match implementation
    config = CommonConfig(config_path="/app.yaml")
    assert config is not None
    assert isinstance(config.get_query_config(), dict)

@pytest.mark.usefixtures("mock_logger")
def test_get_model_config(common_config):
    model = common_config.get_model("llm")
    assert model is not None

