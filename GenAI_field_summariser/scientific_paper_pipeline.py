#!/usr/bin/env python3
"""
Scientific Paper Analysis Pipeline
Optimized for 16GB MacBook Pro with Ollama
"""

import asyncio
import aiohttp
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
import pickle

# Core libraries
import requests
from bs4 import BeautifulSoup
import arxiv
from scholarly import scholarly
import feedparser

# NLP and ML libraries
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import ollama

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Paper:
    """Data structure for scientific papers"""
    title: str
    authors: List[str]
    abstract: str
    url: str
    date: datetime
    venue: str
    keywords: List[str] = None
    full_text: str = ""
    central_questions: List[str] = None
    methods: List[str] = None
    embedding: np.ndarray = None

class PaperCollector:
    """Collects papers from multiple sources"""
    
    def __init__(self, rate_limit: float = 1.0):
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Scientific Research Tool 1.0'
        })
    
    async def collect_arxiv_papers(self, query: str, max_results: int = 100,
                                 days_back: int = 90) -> List[Paper]:
        """Collect papers from arXiv"""
        logger.info(f"Collecting arXiv papers for query: {query}")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in client.results(search):
            if result.published.replace(tzinfo=None) >= start_date:
                paper = Paper(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    url=result.entry_id,
                    date=result.published.replace(tzinfo=None),
                    venue="arXiv",
                    keywords=result.categories
                )
                papers.append(paper)
                
            # Rate limiting
            await asyncio.sleep(self.rate_limit)
        
        logger.info(f"Collected {len(papers)} papers from arXiv")
        return papers
    
    def collect_inspire_papers(self, query: str, max_results: int = 100) -> List[Paper]:
        """Collect papers from INSPIRE-HEP"""
        logger.info(f"Collecting INSPIRE papers for query: {query}")
        
        base_url = "https://inspirehep.net/api/literature"
        params = {
            'q': query,
            'size': min(max_results, 1000),  # API limit
            'sort': 'mostrecent',
            'format': 'json'
        }
        
        try:
            response = self.session.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            papers = []
            for hit in data.get('hits', {}).get('hits', []):
                metadata = hit.get('metadata', {})
                
                # Extract basic info
                title = metadata.get('titles', [{}])[0].get('title', 'No title')
                authors = [author.get('full_name', '')
                          for author in metadata.get('authors', [])]
                
                # Extract abstract
                abstracts = metadata.get('abstracts', [])
                abstract = abstracts[0].get('value', '') if abstracts else ''
                
                # Extract URL
                urls = metadata.get('urls', [])
                url = urls[0].get('value', '') if urls else ''
                
                # Extract date
                date_str = metadata.get('preprint_date',
                                      metadata.get('publication_info', [{}])[0].get('year'))
                try:
                    date = datetime.strptime(str(date_str), '%Y-%m-%d') if '-' in str(date_str) else datetime.strptime(str(date_str), '%Y')
                except:
                    date = datetime.now()
                
                paper = Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=url,
                    date=date,
                    venue="INSPIRE-HEP"
                )
                papers.append(paper)
                
            logger.info(f"Collected {len(papers)} papers from INSPIRE")
            return papers
            
        except Exception as e:
            logger.error(f"Error collecting INSPIRE papers: {e}")
            return []

class TextAnalyzer:
    """Analyzes scientific texts using Ollama and sentence transformers"""
    
    def __init__(self, model_name: str = "llama3.2:3b", embedding_model: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info(f"Initialized TextAnalyzer with {model_name}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for texts in batches"""
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=True)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def extract_central_questions(self, abstract: str, title: str) -> List[str]:
        """Extract central research questions using Ollama"""
        prompt = f"""
        Analyze this scientific paper and identify the main research questions being investigated.
        
        Title: {title}
        Abstract: {abstract}
        
        Please identify 2-3 central research questions that this paper addresses. 
        Format your response as a simple list, one question per line, starting each with "- ".
        Focus on the core scientific questions, not methodological details.
        """
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.3, 'num_predict': 200}
            )
            
            # Parse response
            questions = []
            for line in response['response'].split('\n'):
                line = line.strip()
                if line.startswith('- ') or line.startswith('•'):
                    questions.append(line[2:].strip())
                elif line and not line.startswith('Based on') and '?' in line:
                    questions.append(line.strip())
            
            return questions[:3]  # Limit to top 3
            
        except Exception as e:
            logger.error(f"Error extracting questions: {e}")
            return []
    
    def extract_methods(self, abstract: str, title: str) -> List[str]:
        """Extract research methods using Ollama"""
        prompt = f"""
        Analyze this scientific paper and identify the main research methods and approaches used.
        
        Title: {title}
        Abstract: {abstract}
        
        Please identify the key methods, techniques, or approaches used in this research.
        Format your response as a simple list, one method per line, starting each with "- ".
        Focus on methodological approaches, experimental techniques, or theoretical frameworks.
        """
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.3, 'num_predict': 150}
            )
            
            # Parse response
            methods = []
            for line in response['response'].split('\n'):
                line = line.strip()
                if line.startswith('- ') or line.startswith('•'):
                    methods.append(line[2:].strip())
                elif line and not line.startswith('Based on') and len(line.split()) > 2:
                    methods.append(line.strip())
            
            return methods[:5]  # Limit to top 5
            
        except Exception as e:
            logger.error(f"Error extracting methods: {e}")
            return []
    
    def analyze_papers_batch(self, papers: List[Paper], batch_size: int = 5) -> List[Paper]:
        """Analyze papers in batches to manage memory"""
        logger.info(f"Analyzing {len(papers)} papers in batches of {batch_size}")
        
        analyzed_papers = []
        
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            
            for paper in batch:
                try:
                    # Extract questions and methods
                    paper.central_questions = self.extract_central_questions(
                        paper.abstract, paper.title
                    )
                    paper.methods = self.extract_methods(
                        paper.abstract, paper.title
                    )
                    
                    # Generate embedding
                    text = f"{paper.title} {paper.abstract}"
                    paper.embedding = self.embedding_model.encode([text])[0]
                    
                    analyzed_papers.append(paper)
                    
                    # Rate limiting for Ollama
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error analyzing paper {paper.title}: {e}")
                    analyzed_papers.append(paper)  # Add paper even if analysis fails
            
            logger.info(f"Completed batch {i//batch_size + 1}/{(len(papers)-1)//batch_size + 1}")
        
        return analyzed_papers

class PaperClusterer:
    """Clusters papers based on methods and content"""
    
    def __init__(self, n_neighbors: int = 15, min_cluster_size: int = 3):
        self.n_neighbors = n_neighbors
        self.min_cluster_size = min_cluster_size
        self.umap_model = None
        self.cluster_model = None
        self.reduced_embeddings = None
    
    def cluster_papers(self, papers: List[Paper]) -> Tuple[List[int], np.ndarray]:
        """Cluster papers based on embeddings"""
        logger.info(f"Clustering {len(papers)} papers")
        
        # Extract embeddings
        embeddings = np.array([p.embedding for p in papers if p.embedding is not None])
        valid_papers = [p for p in papers if p.embedding is not None]
        
        if len(embeddings) < self.min_cluster_size:
            logger.warning("Not enough papers with embeddings for clustering")
            return [0] * len(valid_papers), embeddings
        
        # Reduce dimensionality
        self.umap_model = umap.UMAP(
            n_neighbors=min(self.n_neighbors, len(embeddings) - 1),
            n_components=2,
            metric='cosine',
            random_state=42
        )
        self.reduced_embeddings = self.umap_model.fit_transform(embeddings)
        
        # Cluster
        self.cluster_model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='euclidean'
        )
        cluster_labels = self.cluster_model.fit_predict(self.reduced_embeddings)
        
        logger.info(f"Found {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters")
        
        return cluster_labels.tolist(), self.reduced_embeddings
    
    def visualize_clusters(self, papers: List[Paper], cluster_labels: List[int],
                          save_path: Optional[str] = None):
        """Visualize paper clusters"""
        if self.reduced_embeddings is None:
            logger.error("No reduced embeddings available for visualization")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Color by cluster
        scatter = plt.scatter(
            self.reduced_embeddings[:, 0],
            self.reduced_embeddings[:, 1],
            c=cluster_labels,
            cmap='tab10',
            alpha=0.7,
            s=50
        )
        
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('Paper Clusters by Content Similarity')
        plt.colorbar(scatter, label='Cluster')
        
        # Add cluster centroids
        unique_labels = set(cluster_labels)
        for label in unique_labels:
            if label != -1:  # Ignore noise points
                mask = np.array(cluster_labels) == label
                centroid = self.reduced_embeddings[mask].mean(axis=0)
                plt.annotate(f'C{label}', centroid, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class ResultAnalyzer:
    """Analyzes and summarizes results"""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    
    def analyze_research_trends(self, papers: List[Paper]) -> Dict:
        """Analyze research trends across papers"""
        logger.info("Analyzing research trends")
        
        # Collect all questions and methods
        all_questions = []
        all_methods = []
        
        for paper in papers:
            if paper.central_questions:
                all_questions.extend(paper.central_questions)
            if paper.methods:
                all_methods.extend(paper.methods)
        
        # Find common themes
        if all_questions:
            question_text = ' '.join(all_questions)
            question_tfidf = self.tfidf.fit_transform([question_text])
            top_question_terms = [self.tfidf.get_feature_names_out()[i]
                                for i in question_tfidf.toarray()[0].argsort()[-10:]]
        else:
            top_question_terms = []
        
        if all_methods:
            method_text = ' '.join(all_methods)
            method_tfidf = self.tfidf.fit_transform([method_text])
            top_method_terms = [self.tfidf.get_feature_names_out()[i]
                              for i in method_tfidf.toarray()[0].argsort()[-10:]]
        else:
            top_method_terms = []
        
        return {
            'total_papers': len(papers),
            'papers_with_questions': len([p for p in papers if p.central_questions]),
            'papers_with_methods': len([p for p in papers if p.methods]),
            'common_question_themes': top_question_terms[::-1],  # Reverse for descending order
            'common_methods': top_method_terms[::-1],
            'unique_questions': len(set(all_questions)),
            'unique_methods': len(set(all_methods))
        }
    
    def generate_cluster_summaries(self, papers: List[Paper], cluster_labels: List[int]) -> Dict:
        """Generate summaries for each cluster"""
        logger.info("Generating cluster summaries")
        
        cluster_summaries = {}
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
                
            cluster_papers = [papers[i] for i, l in enumerate(cluster_labels) if l == label]
            
            # Collect questions and methods for this cluster
            cluster_questions = []
            cluster_methods = []
            cluster_titles = []
            
            for paper in cluster_papers:
                cluster_titles.append(paper.title)
                if paper.central_questions:
                    cluster_questions.extend(paper.central_questions)
                if paper.methods:
                    cluster_methods.extend(paper.methods)
            
            # Find most common themes
            question_freq = {}
            method_freq = {}
            
            for q in cluster_questions:
                question_freq[q] = question_freq.get(q, 0) + 1
            
            for m in cluster_methods:
                method_freq[m] = method_freq.get(m, 0) + 1
            
            # Sort by frequency
            top_questions = sorted(question_freq.items(), key=lambda x: x[1], reverse=True)[:3]
            top_methods = sorted(method_freq.items(), key=lambda x: x[1], reverse=True)[:3]
            
            cluster_summaries[f"Cluster_{label}"] = {
                'size': len(cluster_papers),
                'top_questions': [q[0] for q in top_questions],
                'top_methods': [m[0] for m in top_methods],
                'sample_titles': cluster_titles[:3]
            }
        
        return cluster_summaries

class ScientificPaperPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, output_dir: str = "output", model_name: str = "llama3.2:3b"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.collector = PaperCollector()
        self.analyzer = TextAnalyzer(model_name=model_name)
        self.clusterer = PaperClusterer()
        self.result_analyzer = ResultAnalyzer()
    
    async def run_pipeline(self, field_query: str, max_papers: int = 50) -> Dict:
        """Run the complete analysis pipeline"""
        logger.info(f"Starting pipeline for field: {field_query}")
        
        # Step 1: Collect papers
        papers = []
        
        # Collect from arXiv
        arxiv_papers = await self.collector.collect_arxiv_papers(
            field_query, max_results=max_papers//2
        )
        papers.extend(arxiv_papers)
        
        # Collect from INSPIRE (if physics-related)
        if any(term in field_query.lower() for term in ['physics', 'quantum', 'particle', 'cosmology']):
            inspire_papers = self.collector.collect_inspire_papers(
                field_query, max_results=max_papers//2
            )
            papers.extend(inspire_papers)
        
        logger.info(f"Collected {len(papers)} total papers")
        
        # Step 2: Analyze papers
        papers = self.analyzer.analyze_papers_batch(papers)
        
        # Step 3: Cluster papers
        valid_papers = [p for p in papers if p.embedding is not None]
        cluster_labels, embeddings = self.clusterer.cluster_papers(valid_papers)
        
        # Step 4: Analyze results
        trends = self.result_analyzer.analyze_research_trends(valid_papers)
        cluster_summaries = self.result_analyzer.generate_cluster_summaries(
            valid_papers, cluster_labels
        )
        
        # Step 5: Generate visualizations
        self.clusterer.visualize_clusters(
            valid_papers, cluster_labels,
            save_path=self.output_dir / "clusters.png"
        )
        
        # Step 6: Save results
        results = {
            'query': field_query,
            'collection_date': datetime.now().isoformat(),
            'trends': trends,
            'cluster_summaries': cluster_summaries,
            'papers': [
                {
                    'title': p.title,
                    'authors': p.authors,
                    'date': p.date.isoformat(),
                    'venue': p.venue,
                    'url': p.url,
                    'central_questions': p.central_questions,
                    'methods': p.methods,
                    'cluster': cluster_labels[i] if i < len(cluster_labels) else -1
                }
                for i, p in enumerate(valid_papers)
            ]
        }
        
        # Save to JSON
        with open(self.output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save papers as pickle for future use
        with open(self.output_dir / "papers.pkl", 'wb') as f:
            pickle.dump(valid_papers, f)
        
        logger.info("Pipeline completed successfully")
        return results
    
    def print_summary(self, results: Dict):
        """Print a summary of results"""
        print("\n" + "="*80)
        print(f"SCIENTIFIC FIELD ANALYSIS: {results['query']}")
        print("="*80)
        
        trends = results['trends']
        print(f"\nCOLLECTION SUMMARY:")
        print(f"   Total papers analyzed: {trends['total_papers']}")
        print(f"   Papers with extracted questions: {trends['papers_with_questions']}")
        print(f"   Papers with extracted methods: {trends['papers_with_methods']}")
        print(f"   Unique research questions identified: {trends['unique_questions']}")
        print(f"   Unique methods identified: {trends['unique_methods']}")
        
        print(f"\n CENTRAL RESEARCH THEMES:")
        for theme in trends['common_question_themes'][:5]:
            print(f"    {theme}")
        
        print(f"\n  COMMON METHODOLOGICAL APPROACHES:")
        for method in trends['common_methods'][:5]:
            print(f"    {method}")
        
        print(f"\n RESEARCH CLUSTERS:")
        for cluster_name, summary in results['cluster_summaries'].items():
            print(f"\n   {cluster_name} ({summary['size']} papers):")
            print(f"   ├─ Key Questions:")
            for q in summary['top_questions'][:2]:
                print(f"   │   {q}")
            print(f"   ├─ Main Methods:")
            for m in summary['top_methods'][:2]:
                print(f"   │   {m}")
            print(f"   └─ Sample Papers:")
            for title in summary['sample_titles']:
                print(f"       {title}")
        
        print("\n" + "="*80)

# Example usage and testing
async def main():
    """Example usage of the pipeline"""
    
    # Initialize pipeline
    pipeline = ScientificPaperPipeline(
        output_dir="physics_ml_analysis",
        model_name="llama3.2:3b"  # Use smaller model for 16GB RAM
    )
    
    # Run analysis
    field_query = "physics inspired machine learning"
    results = await pipeline.run_pipeline(field_query, max_papers=30)
    
    # Print summary
    pipeline.print_summary(results)

if __name__ == "__main__":
    # Check if Ollama is running
    try:
        ollama.list()
        print("Ollama is running")
    except:
        print("Ollama is not running. Please start Ollama first:")
        print("   brew install ollama")
        print("   ollama serve")
        print("   ollama pull llama3.2:3b")
        exit(1)
    
    # Run the pipeline
    asyncio.run(main())
