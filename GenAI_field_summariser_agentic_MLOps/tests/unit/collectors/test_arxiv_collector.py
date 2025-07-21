import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.pipeline.collectors.arxiv_collector import ArxivCollector

@pytest.mark.asyncio
async def test_collect_arxiv_papers(mock_arxiv_client):
    """Test collecting papers from arXiv"""
    collector = ArxivCollector(rate_limit=0.1)  # Fast for testing
    
    # Mock arxiv results
    mock_result = Mock()
    mock_result.title = "Test Paper"
    mock_result.authors = [Mock(name="Test Author")]
    mock_result.summary = "Test abstract"
    mock_result.entry_id = "http://arxiv.org/abs/2301.00001"
    mock_result.published = datetime.now()
    mock_result.categories = ["cs.LG"]
    
    with patch('arxiv.Client') as mock_client_class:
        mock_client = Mock()
        mock_client.results.return_value = [mock_result]
        mock_client_class.return_value = mock_client
        
        papers = await collector.collect_papers("machine learning", max_results=10)
        
        assert len(papers) == 1
        assert papers[0].title == "Test Paper"
        assert papers[0].venue == "arXiv"

@pytest.mark.asyncio 
async def test_rate_limiting(mock_arxiv_client):
    """Test that rate limiting works"""
    collector = ArxivCollector(rate_limit=0.1)
    
    start_time = datetime.now()
    
    with patch('arxiv.Client') as mock_client_class:
        mock_client = Mock()
        mock_client.results.return_value = []
        mock_client_class.return_value = mock_client
        
        await collector.collect_papers("test", max_results=1)
    
    # Should take at least the rate limit time
    elapsed = (datetime.now() - start_time).total_seconds()
    assert elapsed >= 0.1
