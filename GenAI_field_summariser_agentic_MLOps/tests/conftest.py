import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock
from datetime import datetime

from src.pipeline.models.paper import Paper
from src.pipeline.config.config_loader import load_config

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_papers():
    """Create sample papers for testing"""
    return [
        Paper(
            title="Quantum Machine Learning Applications",
            authors=["Alice Smith", "Bob Johnson"],
            abstract="This paper explores the intersection of quantum computing and machine learning...",
            url="http://arxiv.org/abs/2301.00001",
            date=datetime(2023, 1, 1),
            venue="arXiv",
            keywords=["quantum", "machine learning"]
        ),
        Paper(
            title="Neural Networks for Physics Simulation",
            authors=["Charlie Brown"],
            abstract="We present a novel approach using neural networks for physics simulation...",
            url="http://arxiv.org/abs/2301.00002", 
            date=datetime(2023, 1, 2),
            venue="arXiv",
            keywords=["neural networks", "physics"]
        )
    ]

@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response"""
    return {
        'response': """- How can quantum computing enhance machine learning algorithms?
- What are the practical applications of quantum ML?
- How do we handle quantum noise in ML models?"""
    }

@pytest.fixture
def test_config():
    """Load test configuration"""
    return load_config("test")

@pytest.fixture
def mock_arxiv_client():
    """Mock arxiv client"""
    mock = Mock()
    mock.results.return_value = []
    return mock
