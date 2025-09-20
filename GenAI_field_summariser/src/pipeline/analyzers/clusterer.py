# ============================================================================
# src/pipeline/analyzers/clusterer.py
"""
Paper clustering using UMAP and HDBSCAN
"""

import numpy as np
import umap
import hdbscan
from typing import List, Tuple, Optional, Dict, Any
import logging

from ..models.paper import Paper
from ..models.schemas import ClusterConfig
from ..utils.monitoring import PipelineMonitor


class PaperClusterer:
    """Clusters papers based on content similarity"""
    
    def __init__(self, config: ClusterConfig, monitor: Optional[PipelineMonitor] = None):
        self.config = config
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)
        
        # Models will be initialized during clustering
        self.umap_model = None
        self.cluster_model = None
        self.reduced_embeddings = None
        
        # Statistics
        self.stats = {
            'papers_clustered': 0,
            'clusters_found': 0,
            'noise_points': 0,
            'avg_cluster_size': 0.0,
            'silhouette_score': None
        }
    
    def cluster_papers(self, papers: List[Paper]) -> Tuple[List[int], np.ndarray]:
        """Cluster papers based on their embeddings"""
        
        with self.monitor.monitor_operation("paper_clustering") if self.monitor else nullcontext():
            self.logger.info(f"Clustering {len(papers)} papers")
            
            # Extract embeddings from papers that have them
            embeddings = []
            valid_papers = []
            
            for paper in papers:
                if paper.has_embedding:
                    embeddings.append(paper.embedding)
                    valid_papers.append(paper)
            
            embeddings = np.array(embeddings)
            
            if len(embeddings) < self.config.min_cluster_size:
                self.logger.warning(f"Not enough papers with embeddings for clustering: {len(embeddings)}")
                return [0] * len(valid_papers), embeddings
            
            # Reduce dimensionality with UMAP
            self.logger.info("Reducing dimensionality with UMAP")
            self.umap_model = umap.UMAP(
                n_neighbors=min(self.config.n_neighbors, len(embeddings) - 1),
                n_components=self.config.umap_components,
                metric=self.config.metric,
                random_state=self.config.random_state,
                n_jobs=1  # Single thread for stability
            )
            
            self.reduced_embeddings = self.umap_model.fit_transform(embeddings)
            
            # Perform clustering with HDBSCAN
            self.logger.info("Performing clustering with HDBSCAN")
            self.cluster_model = hdbscan.HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                metric='euclidean',
                cluster_selection_epsilon=0.0
            )
            
            cluster_labels = self.cluster_model.fit_predict(self.reduced_embeddings)
            
            # Update paper objects with cluster assignments
            for i, paper in enumerate(valid_papers):
                if i < len(cluster_labels):
                    paper.cluster_id = int(cluster_labels[i])
                    paper.confidence_score = self._calculate_confidence_score(i, cluster_labels[i])
