## src/pipeline/__init__.py
"""
Scientific Paper Analysis Pipeline
Main package initialization
"""

from .models.paper import Paper
from .collectors.base_collector import BaseCollector
from .analyzers.text_analyzer import TextAnalyzer
from .analyzers.clusterer import PaperClusterer
from .analyzers.result_analyzer import ResultAnalyzer
from .pipeline_orchestrator import ScientificPaperPipeline

__version__ = "1.0.0"
__all__ = [
    "Paper",
    "BaseCollector", 
    "TextAnalyzer",
    "PaperClusterer",
    "ResultAnalyzer",
    "ScientificPaperPipeline"
]
