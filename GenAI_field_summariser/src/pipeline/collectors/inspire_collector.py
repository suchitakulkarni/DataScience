# src/pipeline/collectors/arxiv_collector.py
"""
ArXiv paper collector implementation
"""

import asyncio
from typing import List
from datetime import datetime, timedelta
import arxiv

from .base_collector import BaseCollector
from ..models.paper import Paper


class ArxivCollector(BaseCollector):
    """Collector for papers from arXiv"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = arxiv.Client()
    
    def get_source_name(self) -> str:
        return "arxiv"
    
    async def collect_papers(self, query: str, max_results: int = 100) -> List[Paper]:
        """Collect papers from arXiv"""
        self.stats['start_time'] = datetime.now()
        
        with self.monitor.monitor_operation("arxiv_collection", source="arxiv") if self.monitor else nullcontext():
            self.logger.info(f"Collecting arXiv papers for query: {query}")
            
            # Limit max_results to avoid overwhelming the API
            max_results = min(max_results, self.config.max_results_per_source)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config.days_back)
            
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers = []
            try:
                for result in self.client.results(search):
                    self._record_request()
                    
                    # Check date range
                    paper_date = result.published.replace(tzinfo=None)
                    if paper_date < start_date:
                        continue
                    
                    # Create paper object
                    paper = Paper(
                        title=result.title.strip(),
                        authors=[author.name for author in result.authors],
                        abstract=result.summary.strip().replace('\n', ' '),
                        url=result.entry_id,
                        date=paper_date,
                        venue="arXiv",
                        keywords=result.categories if result.categories else []
                    )
                    
                    # Validate paper
                    if self._validate_paper(paper):
                        papers.append(paper)
                    
                    # Rate limiting
                    await self._rate_limit()
                
                self._record_papers_collected(len(papers))
                self.logger.info(f"Successfully collected {len(papers)} papers from arXiv")
                
            except Exception as e:
                self._record_error(e)
                self.logger.error(f"Error collecting from arXiv: {str(e)}")
            
            finally:
                self.stats['end_time'] = datetime.now()
            
            return papers


class nullcontext:
    """Null context manager for when monitor is None"""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
