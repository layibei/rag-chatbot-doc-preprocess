import pytest
from unittest.mock import Mock, patch, mock_open
import os
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from preprocess.loader.loader_factories import DocumentLoaderFactory
from preprocess.loader.pdf_loader import PDFDocLoader
from preprocess.loader.text_loader import TextDocLoader
from preprocess.loader.csv_loader import CSVDocLoader
from preprocess.loader.json_loader import JsonDocLoader
from preprocess.loader.docx_loader import DocxDocLoader


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
                },
                'llm': {
                    'type': 'test',
                    'model': 'test-llm-model'
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
def pdf_loader(mock_common_config):
    return PDFDocLoader()

@pytest.fixture
def text_loader(mock_common_config):
    return TextDocLoader()

@pytest.fixture
def csv_loader(mock_common_config):
    return CSVDocLoader()

@pytest.fixture
def json_loader(mock_common_config):
    return JsonDocLoader()

@pytest.fixture
def docx_loader(mock_common_config):
    return DocxDocLoader()

def test_loader_factory(mock_common_config):
    # Test factory creates correct loader types
    assert isinstance(DocumentLoaderFactory.get_loader('pdf'), PDFDocLoader)
    assert isinstance(DocumentLoaderFactory.get_loader('txt'), TextDocLoader)
    assert isinstance(DocumentLoaderFactory.get_loader('csv'), CSVDocLoader)
    assert isinstance(DocumentLoaderFactory.get_loader('json'), JsonDocLoader)
    assert isinstance(DocumentLoaderFactory.get_loader('docx'), DocxDocLoader)
    
    # Test invalid extension
    with pytest.raises(ValueError):
        DocumentLoaderFactory.get_loader('')
    with pytest.raises(ValueError):
        DocumentLoaderFactory.get_loader('invalid')

@pytest.mark.parametrize("loader_fixture,file_ext", [
    ('pdf_loader', '.pdf'),
    ('text_loader', '.txt'),
    ('csv_loader', '.csv'),
    ('json_loader', '.json'),
    ('docx_loader', '.docx')
])
def test_loader_file_extension_support(loader_fixture, file_ext, request):
    loader = request.getfixturevalue(loader_fixture)
    assert loader.is_supported_file_extension(f'test{file_ext}') == True
    assert loader.is_supported_file_extension('test.wrong') == (loader_fixture == 'text_loader')

@pytest.mark.parametrize("loader_fixture", [
    'pdf_loader',
    'text_loader',
    'csv_loader',
    'docx_loader'
])
def test_loader_get_splitter(loader_fixture, request):
    loader = request.getfixturevalue(loader_fixture)
    splitter = loader.get_splitter("test_file")
    assert isinstance(splitter, RecursiveCharacterTextSplitter)

def test_json_loader_specific(json_loader):
    # Test metadata creation
    with patch('os.path.getctime') as mock_ctime, \
         patch('os.path.getmtime') as mock_mtime:
        mock_ctime.return_value = mock_mtime.return_value = datetime.now().timestamp()
        metadata = json_loader.create_metadata('test.json')
        
        assert 'source' in metadata
        assert 'createdAt' in metadata
        assert 'file_path' in metadata
        assert 'creationDate' in metadata
        assert 'modDate' in metadata

@patch('os.path.exists')
def test_loader_invalid_file(mock_exists, pdf_loader):
    mock_exists.return_value = False
    
    with pytest.raises(ValueError):
        pdf_loader.load('nonexistent.pdf')

@patch('os.path.exists')
def test_loader_empty_file(mock_exists, pdf_loader):
    mock_exists.return_value = True
    
    with patch.object(pdf_loader, 'get_loader') as mock_get_loader:
        mock_loader = Mock()
        mock_loader.load.return_value = []
        mock_get_loader.return_value = mock_loader
        
        with pytest.raises(ValueError):
            pdf_loader.load('empty.pdf') 