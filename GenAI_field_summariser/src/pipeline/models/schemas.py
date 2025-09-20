# src/pipeline/models/schemas.py
"""
Pydantic schemas for configuration validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from pathlib import Path


class CollectorConfig(BaseModel):
    """Configuration for paper collectors"""
    rate_limit: float = Field(default=1.0, ge=0.1, description="Rate limit in seconds")
    max_results_per_source: int = Field(default=100, ge=1, le=1000)
    days_back: int = Field(default=90, ge=1, le=365)
    sources: List[str] = Field(default=["arxiv", "inspire"])
    user_agent: str = "Scientific Research Tool 1.0"
    
    @validator('sources')
    def validate_sources(cls, v):
        valid_sources = {"arxiv", "inspire"}
        invalid = set(v) - valid_sources
        if invalid:
            raise ValueError(f"Invalid sources: {invalid}. Must be from {valid_sources}")
        return v


class AnalyzerConfig(BaseModel):
    """Configuration for text analysis"""
    ollama_model: str = "llama3.2:3b"
    embedding_model: str = "all-MiniLM-L6-v2"
    batch_size: int = Field(default=5, ge=1, le=50)
    max_questions: int = Field(default=3, ge=1, le=10)
    max_methods: int = Field(default=5, ge=1, le=15)
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    ollama_timeout: int = Field(default=30, ge=5, le=120)


class ClusterConfig(BaseModel):
    """Configuration for clustering"""
    n_neighbors: int = Field(default=15, ge=5, le=50)
    min_cluster_size: int = Field(default=3, ge=2, le=20)
    umap_components: int = Field(default=2, ge=2, le=10)
    metric: str = Field(default="cosine", regex="^(cosine|euclidean)$")
    random_state: int = 42


class OutputConfig(BaseModel):
    """Configuration for output generation"""
    save_embeddings: bool = True
    save_visualizations: bool = True
    save_raw_data: bool = True
    output_formats: List[str] = Field(default=["json", "pickle"])
    
    @validator('output_formats')
    def validate_formats(cls, v):
        valid_formats = {"json", "pickle", "csv", "parquet"}
        invalid = set(v) - valid_formats
        if invalid:
            raise ValueError(f"Invalid formats: {invalid}. Must be from {valid_formats}")
        return v


class PipelineConfig(BaseModel):
    """Main pipeline configuration"""
    collector: CollectorConfig = CollectorConfig()
    analyzer: AnalyzerConfig = AnalyzerConfig()
    clusterer: ClusterConfig = ClusterConfig()
    output: OutputConfig = OutputConfig()
    
    # Global settings
    output_dir: str = "output"
    log_level: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR)$")
    max_workers: int = Field(default=4, ge=1, le=16)
    cache_enabled: bool = True
    
    @validator('output_dir')
    def validate_output_dir(cls, v):
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

