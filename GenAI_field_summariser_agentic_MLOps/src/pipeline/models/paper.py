# src/pipeline/models/paper.py
"""
Paper data model and related structures
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import numpy as np


@dataclass
class Paper:
    """Data structure for scientific papers"""
    title: str
    authors: List[str]
    abstract: str
    url: str
    date: datetime
    venue: str
    keywords: Optional[List[str]] = None
    full_text: str = ""
    central_questions: Optional[List[str]] = None
    methods: Optional[List[str]] = None
    embedding: Optional[np.ndarray] = None
    cluster_id: int = -1
    confidence_score: float = 0.0
    
    def __post_init__(self):
        """Validate and normalize data after initialization"""
        if not self.title:
            raise ValueError("Paper title cannot be empty")
        if not self.abstract:
            self.abstract = "No abstract available"
        if self.keywords is None:
            self.keywords = []
        if self.central_questions is None:
            self.central_questions = []
        if self.methods is None:
            self.methods = []
    
    @property
    def is_analyzed(self) -> bool:
        """Check if paper has been analyzed (has questions or methods)"""
        return bool(self.central_questions or self.methods)
    
    @property
    def has_embedding(self) -> bool:
        """Check if paper has embedding vector"""
        return self.embedding is not None
    
    def to_dict(self) -> dict:
        """Convert paper to dictionary for serialization"""
        return {
            'title': self.title,
            'authors': self.authors,
            'abstract': self.abstract,
            'url': self.url,
            'date': self.date.isoformat(),
            'venue': self.venue,
            'keywords': self.keywords,
            'central_questions': self.central_questions,
            'methods': self.methods,
            'cluster_id': self.cluster_id,
            'confidence_score': self.confidence_score,
            'is_analyzed': self.is_analyzed,
            'has_embedding': self.has_embedding
        }

