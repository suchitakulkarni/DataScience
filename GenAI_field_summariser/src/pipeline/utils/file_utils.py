# src/pipeline/utils/file_utils.py
"""
File management utilities
"""

import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Any, List, Dict, Union, Optional
import logging
from datetime import datetime

from ..models.paper import Paper


class FileManager:
    """Manages file I/O operations for the pipeline"""
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save_papers(self, papers: List[Paper], filename: str = "papers") -> Dict[str, Path]:
        """Save papers in multiple formats"""
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as pickle (preserves all data including embeddings)
        pickle_path = self.output_dir / f"{filename}_{timestamp}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(papers, f)
        saved_files['pickle'] = pickle_path
        self.logger.info(f"Saved {len(papers)} papers to {pickle_path}")
        
        # Save as JSON (human-readable, no embeddings)
        json_data = [paper.to_dict() for paper in papers]
        json_path = self.output_dir / f"{filename}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        saved_files['json'] = json_path
        self.logger.info(f"Saved {len(papers)} papers to {json_path}")
        
        # Save as CSV (tabular format)
        csv_path = self.output_dir / f"{filename}_{timestamp}.csv"
        df = pd.DataFrame([
            {
                'title': p.title,
                'authors': '; '.join(p.authors),
                'venue': p.venue,
                'date': p.date,
                'url': p.url,
                'cluster_id': p.cluster_id,
                'has_questions': len(p.central_questions or []) > 0,
                'has_methods': len(p.methods or []) > 0,
                'confidence_score': p.confidence_score
            }
            for p in papers
        ])
        df.to_csv(csv_path, index=False)
        saved_files['csv'] = csv_path
        self.logger.info(f"Saved {len(papers)} papers to {csv_path}")
        
        return saved_files
    
    def load_papers(self, filepath: Union[str, Path]) -> List[Paper]:
        """Load papers from pickle file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                papers = pickle.load(f)
            self.logger.info(f"Loaded {len(papers)} papers from {filepath}")
            return papers
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def save_results(self, results: Dict[str, Any], filename: str = "results") -> Path:
        """Save pipeline results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"{filename}_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved results to {filepath}")
        return filepath
    
    def save_metrics(self, metrics_data: Dict[str, Any], filename: str = "metrics") -> Path:
        """Save metrics data to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"{filename}_{timestamp}.json"
        
        # Convert datetime objects to strings for JSON serialization
        serializable_data = self._make_json_serializable(metrics_data)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        self.logger.info(f"Saved metrics to {filepath}")
        return filepath
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Recursively convert objects to JSON-serializable format"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj
    
    def create_output_structure(self) -> Dict[str, Path]:
        """Create standard output directory structure"""
        subdirs = {
            'data': self.output_dir / 'data',
            'visualizations': self.output_dir / 'visualizations', 
            'metrics': self.output_dir / 'metrics',
            'logs': self.output_dir / 'logs',
            'models': self.output_dir / 'models'
        }
        
        for name, path in subdirs.items():
            path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created output directory: {path}")
        
        return subdirs
