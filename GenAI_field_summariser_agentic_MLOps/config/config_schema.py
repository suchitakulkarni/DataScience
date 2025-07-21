from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
from pathlib import Path

class CollectionConfig(BaseModel):
    rate_limit: float = 1.0
    max_papers_per_source: int = 100
    days_back: int = 90
    sources: List[str] = ["arxiv", "inspire"]

class OllamaConfig(BaseModel):
    model_name: str = "llama3.2:3b"
    temperature: float = 0.3
    num_predict: int = 200
    timeout: int = 30

class EmbeddingConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize_embeddings: bool = True

class ModelsConfig(BaseModel):
    ollama: OllamaConfig = OllamaConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()

class ClusteringConfig(BaseModel):
    n_neighbors: int = 15
    min_cluster_size: int = 3
    umap_components: int = 2
    metric: str = "cosine"

class TextProcessingConfig(BaseModel):
    max_features: int = 100
    stop_words: str = "english"
    min_question_length: int = 10
    max_questions_per_paper: int = 3
    max_methods_per_paper: int = 5

class AnalysisConfig(BaseModel):
    clustering: ClusteringConfig = ClusteringConfig()
    text_processing: TextProcessingConfig = TextProcessingConfig()

class OutputConfig(BaseModel):
    base_dir: str = "output"
    save_embeddings: bool = True
    save_plots: bool = True
    plot_dpi: int = 300

class PipelineConfig(BaseModel):
    collection: CollectionConfig = CollectionConfig()
    models: ModelsConfig = ModelsConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    output: OutputConfig = OutputConfig()
    
    @validator('collection')
    def validate_collection(cls, v):
        if v.rate_limit < 0:
            raise ValueError("rate_limit must be non-negative")
        return v
