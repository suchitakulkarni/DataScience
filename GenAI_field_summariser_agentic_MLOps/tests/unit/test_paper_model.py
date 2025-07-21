import pytest
from datetime import datetime
from src.pipeline.models.paper import Paper

def test_paper_creation():
    """Test basic Paper model creation"""
    paper = Paper(
        title="Test Paper",
        authors=["Author One", "Author Two"],
        abstract="This is a test abstract.",
        url="http://example.com/paper1",
        date=datetime(2023, 1, 1),
        venue="Test Venue"
    )
    
    assert paper.title == "Test Paper"
    assert len(paper.authors) == 2
    assert paper.authors[0] == "Author One"
    assert paper.venue == "Test Venue"
    assert paper.keywords is None
    assert paper.central_questions is None

def test_paper_with_optional_fields():
    """Test Paper with all optional fields"""
    paper = Paper(
        title="Advanced Paper",
        authors=["Author One"],
        abstract="Advanced abstract.",
        url="http://example.com/paper2",
        date=datetime(2023, 2, 1),
        venue="Advanced Venue",
        keywords=["keyword1", "keyword2"],
        full_text="Full text content",
        central_questions=["What is the main question?"],
        methods=["Method A", "Method B"]
    )
    
    assert paper.keywords == ["keyword1", "keyword2"]
    assert paper.central_questions == ["What is the main question?"]
    assert len(paper.methods) == 2
    assert paper.full_text == "Full text content"

def test_paper_string_representation():
    """Test paper string representation"""
    paper = Paper(
        title="String Test Paper",
        authors=["Author"],
        abstract="Abstract",
        url="http://example.com",
        date=datetime(2023, 1, 1),
        venue="Venue"
    )
    
    paper_str = str(paper)
    assert "String Test Paper" in paper_str
    assert "Author" in paper_str
