# src/pipeline/collectors/__init__.py
"""Collectors package initialization"""

from .base_collector import BaseCollector
from .arxiv_collector import ArxivCollector
from .inspire_collector import InspireCollector

__all__ = ["BaseCollector", "ArxivCollector", "InspireCollector"]
