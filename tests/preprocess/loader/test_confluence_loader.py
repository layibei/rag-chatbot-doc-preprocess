import pytest
from preprocess.loader.confluence_loader import ConfluenceLoader

def test_extract_page_id_from_cloud_url():
    loader = ConfluenceLoader()
    url = "https://akjamie.atlassian.net/wiki/spaces/~ragchatbot/pages/66115/How+to+load+web+pages"
    page_id = loader._extract_page_id(url)
    assert page_id == "66115"

def test_extract_page_id_various_formats():
    loader = ConfluenceLoader()
    test_cases = [
        ("https://confluence.example.com/pages/viewpage.action?pageId=123456", "123456"),
        ("https://akjamie.atlassian.net/wiki/spaces/~user/pages/66115/Page+Title", "66115"),
        ("https://confluence.example.com/display/SPACE/Page+Title", None),  # Will return None if API call fails
        ("https://confluence.example.com/pages/123456", "123456"),
    ]
    
    for url, expected_id in test_cases:
        if expected_id is None:
            # Skip API-dependent cases in unit tests
            continue
        page_id = loader._extract_page_id(url)
        assert page_id == expected_id, f"Failed to extract correct page ID from {url}" 