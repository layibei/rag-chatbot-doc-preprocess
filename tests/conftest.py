import os
import pytest
from unittest.mock import Mock, patch
from sqlalchemy import URL

# Mock both loggers before any imports to prevent circular dependency
mock_logger = Mock()
with patch('utils.logger_init.logger', mock_logger), \
     patch('utils.logging_util.logger', mock_logger), \
     patch('utils.logging_util.configure_logger', return_value=mock_logger):
    import dotenv
    from unittest.mock import Mock
    from langchain_core.language_models import BaseChatModel
    from config.common_settings import CommonConfig
    from pathlib import Path
    import yaml

    from config.database.database_manager import DatabaseManager
    from preprocess.index_log import Base as IndexLogBase
    from conversation import Base as ConversationBase

SAMPLE_CONFIG = """
app:
  models:
    llm:
      type: ollama
      model: qwen2.5
      temperature: 0.7
    embedding:
      type: huggingface
      model: sentence-transformers/all-mpnet-base-v2
  query_agent:
    search:
      top_k: 10
      relevance_threshold: 0.7
      rerank_enabled: true
      max_retries: 1
      web_search_enabled: false
    hallucination:
      high_risk: 0.6
      medium_risk: 0.8
    metrics:
      enabled: true
      log_level: INFO
      store_in_db: true
    output:
      generate_suggested_documents: true
"""

@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load environment variables for testing"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    dotenv.load_dotenv(env_path)

@pytest.fixture
def db_manager():
    """Create a database manager with in-memory SQLite for testing"""
    url = URL.create("sqlite", database=":memory:")
    manager = DatabaseManager(url)
    
    # Create all tables
    IndexLogBase.metadata.create_all(manager.engine)
    ConversationBase.metadata.create_all(manager.engine)

    return manager 

@pytest.fixture
def mock_llm():
    llm = Mock(spec=BaseChatModel)
    llm.invoke.return_value = Mock(content="test response")
    return llm

@pytest.fixture
def mock_config():
    config = Mock(spec=CommonConfig)
    config.get_query_config.return_value = True
    return config

@pytest.fixture
def sample_documents():
    from langchain_core.documents import Document
    return [
        Document(page_content="Test content 1", metadata={"source": "test1.txt"}),
        Document(page_content="Test content 2", metadata={"source": "test2.txt"})
    ] 

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

@pytest.fixture
def common_config(tmp_path, monkeypatch, mock_logger):
    # Create a temporary config file
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(SAMPLE_CONFIG)
    
    # Patch BASE_DIR to point to our temp directory
    monkeypatch.setattr('config.common_settings.BASE_DIR', str(tmp_path))
    
    # Add leading slash to match implementation's path handling
    return CommonConfig(config_path="/test_config.yaml") 