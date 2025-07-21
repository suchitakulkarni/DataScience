#!/usr/bin/env python3
"""
Paper data model with enhanced validation and serialization
src/pipeline/models/paper.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np
from pydantic import BaseModel, validator
import json


@dataclass
class Paper:
    """Enhanced data structure for scientific papers with validation"""
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
    cluster_id: Optional[int] = None
    confidence_scores: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate paper data after initialization"""
        if not self.title.strip():
            raise ValueError("Paper title cannot be empty")
        
        if not self.abstract.strip():
            raise ValueError("Paper abstract cannot be empty")
            
        if not self.authors:
            raise ValueError("Paper must have at least one author")
            
        # Clean and validate authors
        self.authors = [author.strip() for author in self.authors if author.strip()]
        
        # Initialize confidence scores if not provided
        if self.confidence_scores is None:
            self.confidence_scores = {}
    
    @property
    def combined_text(self) -> str:
        """Get combined text for embedding generation"""
        text_parts = [self.title, self.abstract]
        
        if self.keywords:
            text_parts.append(" ".join(self.keywords))
            
        if self.full_text:
            text_parts.append(self.full_text[:1000])  # Limit full text
            
        return " ".join(text_parts)
    
    @property
    def has_analysis(self) -> bool:
        """Check if paper has been analyzed (questions + methods extracted)"""
        return bool(self.central_questions and self.methods)
    
    @property  
    def has_embedding(self) -> bool:
        """Check if paper has embedding"""
        return self.embedding is not None
    
    def add_confidence_score(self, metric: str, score: float):
        """Add a confidence score for some analysis metric"""
        if not 0 <= score <= 1:
            raise ValueError(f"Confidence score must be between 0 and 1, got {score}")
        self.confidence_scores[metric] = score
    
    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        """Convert paper to dictionary for serialization"""
        paper_dict = {
            'title': self.title,
            'authors': self.authors,
            'abstract': self.abstract,
            'url': self.url,
            'date': self.date.isoformat(),
            'venue': self.venue,
            'keywords': self.keywords,
            'full_text': self.full_text,
            'central_questions': self.central_questions,
            'methods': self.methods,
            'cluster_id': self.cluster_id,
            'confidence_scores': self.confidence_scores
        }
        
        if include_embedding and self.has_embedding:
            paper_dict['embedding'] = self.embedding.tolist()
            
        return paper_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Paper':
        """Create Paper from dictionary"""
        # Handle date parsing
        if isinstance(data['date'], str):
            data['date'] = datetime.fromisoformat(data['date'].replace('Z', '+00:00')).replace(tzinfo=None)
        
        # Handle embedding
        embedding = None
        if 'embedding' in data and data['embedding'] is not None:
            embedding = np.array(data['embedding'])
            
        return cls(
            title=data['title'],
            authors=data['authors'],
            abstract=data['abstract'],
            url=data['url'],
            date=data['date'],
            venue=data['venue'],
            keywords=data.get('keywords'),
            full_text=data.get('full_text', ''),
            central_questions=data.get('central_questions'),
            methods=data.get('methods'),
            embedding=embedding,
            cluster_id=data.get('cluster_id'),
            confidence_scores=data.get('confidence_scores', {})
        )
    
    def __str__(self) -> str:
        """String representation of paper"""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += f" et al. ({len(self.authors)} total)"
        
        return f"Paper('{self.title}' by {authors_str}, {self.venue}, {self.date.year})"
    
    def __repr__(self) -> str:
        return self.__str__()


class PaperValidationModel(BaseModel):
    """Pydantic model for paper validation"""
    title: str
    authors: List[str]
    abstract: str
    url: str
    date: datetime
    venue: str
    keywords: Optional[List[str]] = None
    
    @validator('title')
    def title_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()
    
    @validator('abstract')
    def abstract_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Abstract cannot be empty')
        if len(v) < 50:
            raise ValueError('Abstract too short (minimum 50 characters)')
        return v.strip()
    
    @validator('authors')
    def authors_not_empty(cls, v):
        if not v:
            raise ValueError('Must have at least one author')
        return [author.strip() for author in v if author.strip()]
    
    @validator('url')
    def url_format(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


def validate_paper_data(paper_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate paper data using Pydantic model"""
    validation_model = PaperValidationModel(**paper_data)
    return validation_model.dict()
