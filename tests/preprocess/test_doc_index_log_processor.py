import pytest
from unittest.mock import Mock, patch
from datetime import datetime, UTC
import hashlib

from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_redis import RedisVectorStore

from preprocess.doc_index_log_processor import DocEmbeddingsProcessor
from preprocess.index_log import Status, IndexLog


@pytest.fixture
def mock_yaml_config():
    return {
        'app': {
            'embedding': {
                'input_path': '/test/input',
                'staging_path': '/test/staging',
                'archive_path': '/test/archive',
                'trunk_size': 1000,
                'overlap': 200
            },
            'models': {
                'embedding': {
                    'type': 'huggingface',
                    'model': 'test-embedding-model'
                }
            }
        }
    }

@pytest.fixture
def mock_common_config(mock_yaml_config):
    with patch('config.common_settings.CommonConfig.load_yaml_file', return_value=mock_yaml_config), \
         patch('config.common_settings.os.path.exists', return_value=True), \
         patch('config.common_settings.dotenv.load_dotenv'), \
         patch('config.common_settings.BASE_DIR', '/test'):
        from config.common_settings import CommonConfig
        return CommonConfig()

@pytest.fixture
def mock_embeddings():
    embeddings = Mock()
    embeddings.dimension = 1536  # Standard OpenAI embedding dimension
    return embeddings

@pytest.fixture
def mock_vector_store():
    store = Mock()
    store.similarity_search_with_score = Mock(return_value=[
        (Document(page_content="test", metadata={"id": "doc1"}), 0.8)
    ])
    store.delete = Mock()
    return store

@pytest.fixture
def mock_index_log_helper():
    helper = Mock()
    helper.find_by_checksum.return_value = None
    helper.find_by_source.return_value = None
    helper.create.return_value = IndexLog(
        id="test_id",
        source="test_source",
        source_type="pdf",
        checksum="test_checksum",
        status=Status.PENDING,
        created_at=datetime.now(UTC),
        modified_at=datetime.now(UTC),
        created_by="test_user",
        modified_by="test_user"
    )
    return helper

@pytest.fixture
def processor(mock_embeddings, mock_vector_store, mock_index_log_helper, mock_common_config):
    return DocEmbeddingsProcessor(mock_embeddings, mock_vector_store, mock_index_log_helper)

def test_get_source_type(processor):
    assert processor._get_source_type('pdf') == 'pdf'
    assert processor._get_source_type('txt') == 'text'
    assert processor._get_source_type('csv') == 'csv'
    assert processor._get_source_type('json') == 'json'
    assert processor._get_source_type('docx') == 'docx'
    assert processor._get_source_type('unknown') is None

@patch('builtins.open', create=True)
def test_calculate_checksum(mock_open, processor):
    mock_open.return_value.__enter__.return_value.read.return_value = b'test content'
    expected_checksum = hashlib.sha256(b'test content').hexdigest()
    
    result = processor._calculate_checksum('test_file.pdf')
    assert result == expected_checksum

def test_add_index_log_new_document(processor, mock_index_log_helper):
    with patch.object(processor, '_calculate_checksum') as mock_calc:
        mock_calc.return_value = 'new_checksum'
        
        result = processor.add_index_log(
            source='test_file.pdf',
            source_type='pdf',
            user_id='test_user'
        )
        
        assert result['message'] == 'Document queued for processing'
        assert result['source'] == 'test_file.pdf'
        assert result['source_type'] == 'pdf'
        mock_index_log_helper.create.assert_called_once()

def test_add_index_log_existing_document(processor, mock_index_log_helper):
    existing_log = IndexLog(
        id="existing_id",
        source="test_file.pdf",
        source_type="pdf",
        checksum="existing_checksum",
        status=Status.COMPLETED,
        created_at=datetime.now(UTC),
        modified_at=datetime.now(UTC),
        created_by="test_user",
        modified_by="test_user"
    )
    mock_index_log_helper.find_by_checksum.return_value = existing_log
    
    with patch.object(processor, '_calculate_checksum') as mock_calc:
        mock_calc.return_value = 'existing_checksum'
        
        result = processor.add_index_log(
            source='test_file.pdf',
            source_type='pdf',
            user_id='test_user'
        )
        
        assert result['message'] == 'Document with same content already exists'
        assert result['source'] == 'test_file.pdf'
        mock_index_log_helper.create.assert_not_called()

def test_remove_existing_embeddings_redis(processor):
    # Create a mock Redis store with the necessary methods
    redis_store = Mock()
    # Make the mock appear as RedisVectorStore
    redis_store.__class__ = RedisVectorStore
    redis_store.search_by_metadata = Mock(return_value=[
        Document(page_content="test", metadata={"id": "doc1"})
    ])
    redis_store.delete = Mock()
    processor.vector_store = redis_store
    
    processor._remove_existing_embeddings("test_source", "test_checksum")
    
    redis_store.search_by_metadata.assert_called_once_with({
        "source": "test_source",
        "checksum": "test_checksum"
    })
    redis_store.delete.assert_called_once_with(["doc1"])

def test_remove_existing_embeddings_pgvector(processor, mock_embeddings):
    # Create a mock PGVector store with the necessary methods
    pg_store = Mock()
    # Make the mock appear as PGVector
    pg_store.__class__ = PGVector
    pg_store.similarity_search = Mock(return_value=[
        Document(page_content="test", metadata={"id": "doc1"})
    ])
    pg_store.delete = Mock()
    processor.vector_store = pg_store
    
    processor._remove_existing_embeddings("test_source", "test_type", "test_checksum")
    
    # Verify the correct method was called with appropriate parameters
    pg_store.similarity_search.assert_called_once_with(
        query="",
        k=100,
        filter={
            "source": "test_source",
            "source_type": "test_type",
            "checksum": "test_checksum"
        }
    )
    pg_store.delete.assert_called_once_with(["doc1"])

def test_remove_existing_embeddings_unsupported_store(processor):
    # Test with an unsupported vector store type
    unsupported_store = Mock()
    unsupported_store.delete = Mock()
    processor.vector_store = unsupported_store
    
    # Mock the logger to avoid formatting issues
    mock_logger = Mock()
    processor.logger = mock_logger
    
    # Should not raise an error, but log a warning
    processor._remove_existing_embeddings("test_source", "test_checksum")
    
    # Verify warning was logged
    mock_logger.warning.assert_called_once()
    assert "Unsupported vector store type" in mock_logger.warning.call_args[0][0]
    
    # Verify delete was not called
    unsupported_store.delete.assert_not_called()

def test_get_document_by_id(processor, mock_index_log_helper):
    expected_log = IndexLog(
        id="test_id",
        source="test_source",
        source_type="pdf",
        checksum="test_checksum",
        status=Status.COMPLETED,
        created_at=datetime.now(UTC),
        modified_at=datetime.now(UTC),
        created_by="test_user",
        modified_by="test_user"
    )
    mock_index_log_helper.find_by_id.return_value = expected_log
    
    result = processor.get_document_by_id("test_id")
    
    assert result == expected_log
    mock_index_log_helper.find_by_id.assert_called_once_with("test_id") 