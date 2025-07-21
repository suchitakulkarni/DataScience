# src/pipeline/collectors/base_collector.py
"""
Base collector abstract class
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import asyncio
import logging
from datetime import datetime, timedelta

from ..models.paper import Paper
from ..models.schemas import CollectorConfig
from ..utils.monitoring import PipelineMonitor


class BaseCollector(ABC):
    """Abstract base class for paper collectors"""
    
    def __init__(self, config: CollectorConfig, monitor: Optional[PipelineMonitor] = None):
        self.config = config
        self.monitor = monitor
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Rate limiting
        self._last_request_time = 0.0
        
        # Statistics
        self.stats = {
            'papers_collected': 0,
            'requests_made': 0,
            'errors_encountered': 0,
            'start_time': None,
            'end_time': None
        }
    
    @abstractmethod
    async def collect_papers(self, query: str, max_results: int = 100) -> List[Paper]:
        """Collect papers from the source"""
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Return the name of this paper source"""
        pass
    
    async def _rate_limit(self):
        """Apply rate limiting between requests"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.config.rate_limit:
            sleep_time = self.config.rate_limit - time_since_last
            await asyncio.sleep(sleep_time)
        
        self._last_request_time = asyncio.get_event_loop().time()
    
    def _record_request(self):
        """Record that a request was made"""
        self.stats['requests_made'] += 1
        if self.monitor:
            self.monitor.record_data_point(
                f"{self.get_source_name()}.requests",
                1,
                source=self.get_source_name()
            )
    
    def _record_error(self, error: Exception):
        """Record that an error occurred"""
        self.stats['errors_encountered'] += 1
        self.logger.error(f"Error in {self.get_source_name()}: {str(error)}")
        
        if self.monitor:
            self.monitor.record_data_point(
                f"{self.get_source_name()}.errors",
                1,
                source=self.get_source_name(),
                error_type=type(error).__name__
            )
    
    def _record_papers_collected(self, count: int):
        """Record number of papers collected"""
        self.stats['papers_collected'] += count
        
        if self.monitor:
            self.monitor.record_data_point(
                f"{self.get_source_name()}.papers_collected",
                count,
                source=self.get_source_name()
            )
    
    def _validate_paper(self, paper: Paper) -> bool:
        """Validate that a paper meets minimum requirements"""
        if not paper.title or not paper.title.strip():
            return False
        if not paper.abstract or not paper.abstract.strip():
            return False
        if not paper.authors:
            return False
        return True
    
    def _is_within_date_range(self, paper_date: datetime) -> bool:
        """Check if paper is within the configured date range"""
        cutoff_date = datetime.now() - timedelta(days=self.config.days_back)
        return paper_date >= cutoff_date
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        stats = self.stats.copy()
        if stats['start_time'] and stats['end_time']:
            stats['duration_seconds'] = (stats['end_time'] - stats['start_time']).total_seconds()
            if stats['duration_seconds'] > 0:
                stats['papers_per_second'] = stats['papers_collected'] / stats['duration_seconds']
        return stats
    
    def reset_stats(self):
        """Reset collection statistics"""
        self.stats = {
            'papers_collected': 0,
            'requests_made': 0,
            'errors_encountered': 0,
            'start_time': None,
            'end_time': None
        }

