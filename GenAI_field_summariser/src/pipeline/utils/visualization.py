# src/pipeline/utils/visualization.py
"""
Visualization utilities for pipeline results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from wordcloud import WordCloud

from ..models.paper import Paper


class Visualizer:
    """Creates visualizations for pipeline results"""
    
    def __init__(self, output_dir: Optional[Path] = None, style: str = 'seaborn-v0_8'):
        self.output_dir = Path(output_dir) if output_dir else Path('visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use(style if style in plt.style.available else 'default')
        self.logger = logging.getLogger(__name__)
        
        # Set up color palette
        self.colors = sns.color_palette("husl", 12)
        self.figure_size = (12, 8)
    
    def plot_cluster_scatter(self, 
                           embeddings: np.ndarray,
                           cluster_labels: List[int],
                           papers: List[Paper],
                           title: str = "Paper Clusters",
                           save_path: Optional[str] = None) -> Path:
        """Create scatter plot of paper clusters"""
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create scatter plot
        scatter = ax.scatter(
            embeddings[:, 0], 
            embeddings[:, 1],
            c=cluster_labels,
            cmap='tab10',
            alpha=0.7,
            s=60,
            edgecolors='white',
            linewidth=0.5
        )
        
        # Customize plot
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12) 
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Cluster ID')
        
        # Add cluster centroids and labels
        unique_labels = set(cluster_labels)
        for label in unique_labels:
            if label != -1:  # Skip noise points
                mask = np.array(cluster_labels) == label
                if np.any(mask):
                    centroid = embeddings[mask].mean(axis=0)
                    ax.annotate(
                        f'C{label}', 
                        centroid,
                        fontsize=10, 
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
                    )
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / 'cluster_scatter.png'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved cluster scatter plot to {save_path}")
        return save_path
    
    def plot_papers_timeline(self, 
                           papers: List[Paper],
                           title: str = "Papers by Publication Date",
                           save_path: Optional[str] = None) -> Path:
        """Create timeline visualization of paper publications"""
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Extract dates and venues
        dates = [p.date for p in papers]
        venues = [p.venue for p in papers]
        
        # Create histogram by month
        ax.hist(dates, bins=20, alpha=0.7, color=self.colors[0], edgecolor='black')
        
        ax.set_xlabel('Publication Date', fontsize=12)
        ax.set_ylabel('Number of Papers', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / 'papers_timeline.png'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved timeline plot to {save_path}")
        return save_path
    
    def plot_venue_distribution(self,
                              papers: List[Paper], 
                              title: str = "Papers by Venue",
                              save_path: Optional[str] = None) -> Path:
        """Create bar plot of paper distribution by venue"""
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Count papers by venue
        venue_counts = {}
        for paper in papers:
            venue_counts[paper.venue] = venue_counts.get(paper.venue, 0) + 1
        
        # Sort by count
        sorted_venues = sorted(venue_counts.items(), key=lambda x: x[1], reverse=True)
        venues, counts = zip(*sorted_venues) if sorted_venues else ([], [])
        
        # Create bar plot
        bars = ax.bar(venues, counts, color=self.colors[:len(venues)], alpha=0.8)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Venue', fontsize=12)
        ax.set_ylabel('Number of Papers', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if needed
        if len(venues) > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / 'venue_distribution.png'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved venue distribution plot to {save_path}")
        return save_path
    
    def create_wordcloud(self,
                        texts: List[str],
                        title: str = "Word Cloud",
                        save_path: Optional[str] = None,
                        max_words: int = 100) -> Path:
        """Create word cloud from text data"""
        
        if not texts:
            self.logger.warning("No texts provided for word cloud")
            return None
        
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            max_words=max_words,
            colormap='viridis',
            relative_scaling=0.5
        ).generate(combined_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / 'wordcloud.png'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved word cloud to {save_path}")
        return save_path
    
    def plot_cluster_sizes(self,
                          cluster_labels: List[int],
                          title: str = "Cluster Size Distribution",
                          save_path: Optional[str] = None) -> Path:
        """Plot distribution of cluster sizes"""
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Count cluster sizes (excluding noise -1)
        cluster_sizes = {}
        for label in cluster_labels:
            if label != -1:
                cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
        
        if not cluster_sizes:
            self.logger.warning("No valid clusters found")
            return None
        
        # Sort by cluster ID
        sorted_clusters = sorted(cluster_sizes.items())
        cluster_ids, sizes = zip(*sorted_clusters)
        
        # Create bar plot
        bars = ax.bar([f'C{cid}' for cid in cluster_ids], sizes, 
                     color=self.colors[:len(cluster_ids)], alpha=0.8)
        
        # Add value labels
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{size}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Number of Papers', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / 'cluster_sizes.png'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved cluster sizes plot to {save_path}")
        return save_path

