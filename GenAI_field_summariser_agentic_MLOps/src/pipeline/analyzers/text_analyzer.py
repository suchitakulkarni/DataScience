# src/pipeline/analyzers/text_analyzer.py
"""
Text analysis using Ollama and sentence transformers
"""

import time
import asyncio
import ollama
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import logging

from ..models.paper import Paper
from ..models.schemas import AnalyzerConfig
from ..utils.monitoring import PipelineMonitor


class TextAnalyzer:
    """Analyzes scientific texts using Ollama and sentence transformers"""
    
    def __init__(self, config: AnalyzerConfig, monitor: Optional[PipelineMonitor] = None):
        self.config = config
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(config.embedding_model)
            self.logger.info(f"Loaded embedding model: {config.embedding_model}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Verify Ollama connection
        try:
            ollama.list()
            self.logger.info(f"Connected to Ollama with model: {config.ollama_model}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama: {e}")
            raise
        
        # Statistics
        self.stats = {
            'papers_analyzed': 0,
            'questions_extracted': 0,
            'methods_extracted': 0,
            'embeddings_generated': 0,
            'ollama_calls': 0,
            'ollama_errors': 0
        }
    
    async def analyze_papers_batch(self, papers: List[Paper]) -> List[Paper]:
        """Analyze papers in batches to manage memory and rate limits"""
        self.logger.info(f"Analyzing {len(papers)} papers in batches of {self.config.batch_size}")
        
        analyzed_papers = []
        
        for i in range(0, len(papers), self.config.batch_size):
            batch = papers[i:i + self.config.batch_size]
            
            with self.monitor.monitor_operation("analyze_batch", batch_size=len(batch)) if self.monitor else nullcontext():
                batch_analyzed = await self._analyze_batch(batch)
                analyzed_papers.extend(batch_analyzed)
                
                self.logger.info(f"Completed batch {i//self.config.batch_size + 1}/{(len(papers)-1)//self.config.batch_size + 1}")
                
                # Small delay between batches to prevent overwhelming Ollama
                if i + self.config.batch_size < len(papers):
                    await asyncio.sleep(1)
        
        self.logger.info(f"Analysis complete. Processed {len(analyzed_papers)} papers")
        return analyzed_papers
    
    async def _analyze_batch(self, batch: List[Paper]) -> List[Paper]:
        """Analyze a single batch of papers"""
        analyzed_batch = []
        
        for paper in batch:
            try:
                analyzed_paper = await self._analyze_single_paper(paper)
                analyzed_batch.append(analyzed_paper)
                self.stats['papers_analyzed'] += 1
                
            except Exception as e:
                self.logger.error(f"Error analyzing paper '{paper.title}': {str(e)}")
                # Add paper even if analysis fails
                analyzed_batch.append(paper)
        
        return analyzed_batch
    
    async def _analyze_single_paper(self, paper: Paper) -> Paper:
        """Analyze a single paper"""
        # Extract questions and methods using Ollama
        try:
            questions_task = self._extract_central_questions(paper.abstract, paper.title)
            methods_task = self._extract_methods(paper.abstract, paper.title)
            
            # Run both extractions concurrently
            paper.central_questions, paper.methods = await asyncio.gather(
                questions_task, methods_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(paper.central_questions, Exception):
                self.logger.warning(f"Failed to extract questions: {paper.central_questions}")
                paper.central_questions = []
            
            if isinstance(paper.methods, Exception):
                self.logger.warning(f"Failed to extract methods: {paper.methods}")
                paper.methods = []
            
        except Exception as e:
            self.logger.error(f"Error in Ollama extraction: {e}")
            paper.central_questions = []
            paper.methods = []
        
        # Generate embedding
        try:
            text = f"{paper.title} {paper.abstract}"
            paper.embedding = await self._generate_embedding(text)
            self.stats['embeddings_generated'] += 1
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            paper.embedding = None
        
        return paper
    
    async def _extract_central_questions(self, abstract: str, title: str) -> List[str]:
        """Extract central research questions using Ollama"""
        prompt = self._build_questions_prompt(title, abstract)
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: ollama.generate(
                    model=self.config.ollama_model,
                    prompt=prompt,
                    options={
                        'temperature': self.config.temperature,
                        'num_predict': 200
                    }
                )
            )
            
            self.stats['ollama_calls'] += 1
            questions = self._parse_list_response(response['response'])
            self.stats['questions_extracted'] += len(questions)
            
            return questions[:self.config.max_questions]
            
        except Exception as e:
            self.stats['ollama_errors'] += 1
            self.logger.error(f"Error extracting questions with Ollama: {e}")
            return []
    
    async def _extract_methods(self, abstract: str, title: str) -> List[str]:
        """Extract research methods using Ollama"""
        prompt = self._build_methods_prompt(title, abstract)
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ollama.generate(
                    model=self.config.ollama_model,
                    prompt=prompt,
                    options={
                        'temperature': self.config.temperature,
                        'num_predict': 150
                    }
                )
            )
            
            self.stats['ollama_calls'] += 1
            methods = self._parse_list_response(response['response'])
            self.stats['methods_extracted'] += len(methods)
            
            return methods[:self.config.max_methods]
            
        except Exception as e:
            self.stats['ollama_errors'] += 1
            self.logger.error(f"Error extracting methods with Ollama: {e}")
            return []
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        def encode_text():
            return self.embedding_model.encode([text])[0]
        
        # Run embedding generation in thread pool to avoid blocking
        embedding = await asyncio.get_event_loop().run_in_executor(None, encode_text)
        return embedding
    
    def _build_questions_prompt(self, title: str, abstract: str) -> str:
        """Build prompt for extracting research questions"""
        return f"""
Analyze this scientific paper and identify the main research questions being investigated.

Title: {title}
Abstract: {abstract}

Please identify 2-3 central research questions that this paper addresses.
Format your response as a simple list, one question per line, starting each with "- ".
Focus on the core scientific questions, not methodological details.
Be concise and specific.
"""
    
    def _build_methods_prompt(self, title: str, abstract: str) -> str:
        """Build prompt for extracting research methods"""
        return f"""
Analyze this scientific paper and identify the main research methods and approaches used.

Title: {title}
Abstract: {abstract}

Please identify the key methods, techniques, or approaches used in this research.
Format your response as a simple list, one method per line, starting each with "- ".
Focus on methodological approaches, experimental techniques, or theoretical frameworks.
Be specific and concise.
"""
    
    def _parse_list_response(self, response: str) -> List[str]:
        """Parse Ollama response into a list of items"""
        items = []
        
        for line in response.split('\n'):
            line = line.strip()
            
            # Look for lines starting with bullet points
            if line.startswith('- ') or line.startswith('• '):
                items.append(line[2:].strip())
            # Look for numbered lists
            elif line and line[0].isdigit() and '. ' in line:
                items.append(line.split('. ', 1)[1].strip())
            # Look for other meaningful lines (not headers or filler)
            elif (line and 
                  not line.startswith('Based on') and 
                  not line.startswith('The ') and
                  len(line.split()) > 3 and
                  not line.endswith(':')):
                items.append(line.strip())
        
        # Clean up items
        cleaned_items = []
        for item in items:
            if item and len(item) > 10:  # Filter out very short items
                cleaned_items.append(item)
        
        return cleaned_items
    
    def get_stats(self) -> dict:
        """Get analysis statistics"""
        return self.stats.copy()


class nullcontext:
    """Null context manager for when monitor is None"""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
