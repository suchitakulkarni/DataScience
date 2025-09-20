#!/usr/bin/env python3
"""
Comprehensive Evaluation Framework for Scientific Paper Analysis Pipeline
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime
import asyncio

from scientific_paper_pipeline import ScientificPaperPipeline, Paper

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    # Content Quality Metrics
    question_extraction_rate: float
    method_extraction_rate: float
    question_coherence_score: float
    method_coherence_score: float
    
    # Clustering Quality Metrics
    silhouette_score: float
    cluster_cohesion_score: float
    cluster_separation_score: float
    noise_ratio: float
    
    # Coverage Metrics
    field_coverage_score: float
    temporal_coverage_score: float
    venue_diversity_score: float
    
    # Efficiency Metrics
    processing_time_per_paper: float
    memory_efficiency_score: float
    api_success_rate: float

class PipelineEvaluator:
    """Comprehensive pipeline evaluation system"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.ground_truth_data = {}
        self.benchmark_results = {}
    
    def create_ground_truth_dataset(self, field: str, expert_annotations: Optional[Dict] = None):
        """
        Create or load ground truth dataset for evaluation
        This would ideally be created by domain experts
        """
        logger.info(f"Creating ground truth dataset for {field}")
        
        # Example ground truth structure - would be manually annotated by experts
        ground_truth = {
            "field": field,
            "expert_questions": [
                "How can quantum mechanics principles improve machine learning algorithms?",
                "What are the fundamental limitations of physics-inspired neural networks?",
                "How do symmetry constraints affect model performance?"
            ],
            "expert_methods": [
                "Variational quantum circuits",
                "Physics-informed neural networks",
                "Hamiltonian neural networks",
                "Graph neural networks with physical constraints"
            ],
            "expert_clusters": {
                "quantum_ml": ["quantum neural networks", "variational quantum", "quantum advantage"],
                "physics_informed": ["physics informed", "differential equations", "conservation laws"],
                "geometric_learning": ["geometric deep learning", "symmetry", "group theory"]
            },
            "key_papers": [
                {
                    "title": "Physics-informed neural networks: A deep learning framework...",
                    "expected_cluster": "physics_informed",
                    "key_methods": ["physics informed neural networks", "automatic differentiation"],
                    "central_question": "How to incorporate physical laws into neural networks?"
                }
            ]
        }
        
        if expert_annotations:
            ground_truth.update(expert_annotations)
        
        # Save ground truth
        with open(self.output_dir / f"ground_truth_{field.replace(' ', '_')}.json", 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        self.ground_truth_data[field] = ground_truth
        return ground_truth
    
    def evaluate_content_extraction(self, papers: List[Paper], ground_truth: Dict) -> Dict[str, float]:
        """Evaluate quality of question and method extraction"""
        logger.info("Evaluating content extraction quality")
        
        # Calculate extraction rates
        papers_with_questions = len([p for p in papers if p.central_questions])
        papers_with_methods = len([p for p in papers if p.methods])
        
        question_extraction_rate = papers_with_questions / len(papers) if papers else 0
        method_extraction_rate = papers_with_methods / len(papers) if papers else 0
        
        # Evaluate coherence with expert knowledge
        all_extracted_questions = []
        all_extracted_methods = []
        
        for paper in papers:
            if paper.central_questions:
                all_extracted_questions.extend(paper.central_questions)
            if paper.methods:
                all_extracted_methods.extend(paper.methods)
        
        # Calculate semantic similarity with ground truth
        question_coherence = self._calculate_semantic_overlap(
            all_extracted_questions,
            ground_truth.get("expert_questions", [])
        )
        
        method_coherence = self._calculate_semantic_overlap(
            all_extracted_methods,
            ground_truth.get("expert_methods", [])
        )
        
        return {
            "question_extraction_rate": question_extraction_rate,
            "method_extraction_rate": method_extraction_rate,
            "question_coherence_score": question_coherence,
            "method_coherence_score": method_coherence
        }
    
    def evaluate_clustering_quality(self, papers: List[Paper], cluster_labels: List[int],
                                   ground_truth: Dict) -> Dict[str, float]:
        """Evaluate clustering quality using multiple metrics"""
        logger.info("Evaluating clustering quality")
        
        # Filter papers with embeddings
        valid_papers = [p for p in papers if p.embedding is not None]
        valid_labels = cluster_labels[:len(valid_papers)]
        embeddings = np.array([p.embedding for p in valid_papers])
        
        if len(embeddings) < 3:
            return {"error": "Not enough papers for clustering evaluation"}
        
        # Silhouette score
        silhouette_avg = silhouette_score(embeddings, valid_labels)
        
        # Cluster cohesion and separation
        cohesion_scores = []
        separation_scores = []
        
        unique_labels = set(valid_labels)
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
                
            cluster_mask = np.array(valid_labels) == label
            cluster_embeddings = embeddings[cluster_mask]
            
            if len(cluster_embeddings) > 1:
                # Intra-cluster similarity (cohesion)
                intra_sim = cosine_similarity(cluster_embeddings).mean()
                cohesion_scores.append(intra_sim)
                
                # Inter-cluster distance (separation)
                other_embeddings = embeddings[~cluster_mask]
                if len(other_embeddings) > 0:
                    inter_sim = cosine_similarity(cluster_embeddings, other_embeddings).mean()
                    separation_scores.append(1 - inter_sim)  # Convert similarity to distance
        
        cluster_cohesion = np.mean(cohesion_scores) if cohesion_scores else 0
        cluster_separation = np.mean(separation_scores) if separation_scores else 0
        
        # Noise ratio
        noise_count = sum(1 for label in valid_labels if label == -1)
        noise_ratio = noise_count / len(valid_labels) if valid_labels else 0
        
        return {
            "silhouette_score": silhouette_avg,
            "cluster_cohesion_score": cluster_cohesion,
            "cluster_separation_score": cluster_separation,
            "noise_ratio": noise_ratio
        }
    
    def evaluate_field_coverage(self, papers: List[Paper], field: str) -> Dict[str, float]:
        """Evaluate how well the pipeline covers the scientific field"""
        logger.info("Evaluating field coverage")
        
        # Temporal coverage
        if papers:
            dates = [p.date for p in papers if p.date]
            if dates:
                date_range = (max(dates) - min(dates)).days
                temporal_coverage = min(date_range / 365.0, 1.0)  # Normalize to [0,1]
            else:
                temporal_coverage = 0
        else:
            temporal_coverage = 0
        
        # Venue diversity
        venues = [p.venue for p in papers if p.venue]
        venue_diversity = len(set(venues)) / len(venues) if venues else 0
        
        # Author diversity (as proxy for field breadth)
        all_authors = []
        for paper in papers:
            all_authors.extend(paper.authors)
        
        author_diversity = len(set(all_authors)) / len(all_authors) if all_authors else 0
        
        # Keyword coverage (simplified field coverage metric)
        field_keywords = self._extract_field_keywords(field)
        paper_text = []
        for paper in papers:
            paper_text.append(f"{paper.title} {paper.abstract}")
        
        keyword_coverage = self._calculate_keyword_coverage(paper_text, field_keywords)
        
        return {
            "temporal_coverage_score": temporal_coverage,
            "venue_diversity_score": venue_diversity,
            "author_diversity_score": author_diversity,
            "field_coverage_score": keyword_coverage
        }
    
    def benchmark_against_baselines(self, papers: List[Paper], cluster_labels: List[int]) -> Dict:
        """Compare against baseline methods"""
        logger.info("Running baseline comparisons")
        
        embeddings = np.array([p.embedding for p in papers if p.embedding is not None])
        
        if len(embeddings) < 3:
            return {"error": "Not enough data for benchmarking"}
        
        # Baseline 1: Random clustering
        random_labels = np.random.randint(0, max(cluster_labels) + 1, len(embeddings))
        random_silhouette = silhouette_score(embeddings, random_labels)
        
        # Baseline 2: Simple K-means
        from sklearn.cluster import KMeans
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        if n_clusters > 0:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_labels = kmeans.fit_predict(embeddings)
            kmeans_silhouette = silhouette_score(embeddings, kmeans_labels)
        else:
            kmeans_silhouette = -1
        
        # Our method
        valid_labels = [l for l in cluster_labels if l != -1]
        if len(valid_labels) > 1:
            our_silhouette = silhouette_score(
                embeddings[[i for i, l in enumerate(cluster_labels) if l != -1]],
                valid_labels
            )
        else:
            our_silhouette = -1
        
        return {
            "random_baseline_silhouette": random_silhouette,
            "kmeans_baseline_silhouette": kmeans_silhouette,
            "our_method_silhouette": our_silhouette,
            "improvement_over_random": our_silhouette - random_silhouette,
            "improvement_over_kmeans": our_silhouette - kmeans_silhouette
        }
    
    def evaluate_llm_quality(self, papers: List[Paper], sample_size: int = 10) -> Dict:
        """Evaluate LLM extraction quality through manual inspection"""
        logger.info("Evaluating LLM extraction quality")
        
        # Sample papers for detailed evaluation
        sample_papers = np.random.choice(papers, min(sample_size, len(papers)), replace=False)
        
        quality_scores = []
        
        for paper in sample_papers:
            # Quality metrics for each paper
            paper_score = {
                "title": paper.title,
                "has_questions": bool(paper.central_questions),
                "has_methods": bool(paper.methods),
                "question_count": len(paper.central_questions) if paper.central_questions else 0,
                "method_count": len(paper.methods) if paper.methods else 0,
            }
            
            # Simple quality heuristics
            if paper.central_questions:
                # Check if questions are actually questions
                question_quality = sum(1 for q in paper.central_questions if '?' in q or 'how' in q.lower() or 'what' in q.lower()) / len(paper.central_questions)
                paper_score["question_quality"] = question_quality
            
            if paper.methods:
                # Check if methods seem methodological (contain certain keywords)
                method_keywords = ['algorithm', 'method', 'approach', 'technique', 'model', 'framework']
                method_quality = sum(1 for m in paper.methods if any(kw in m.lower() for kw in method_keywords)) / len(paper.methods)
                paper_score["method_quality"] = method_quality
            
            quality_scores.append(paper_score)
        
        # Aggregate scores
        avg_question_quality = np.mean([s.get("question_quality", 0) for s in quality_scores])
        avg_method_quality = np.mean([s.get("method_quality", 0) for s in quality_scores])
        
        return {
            "sample_size": len(sample_papers),
            "avg_question_quality": avg_question_quality,
            "avg_method_quality": avg_method_quality,
            "detailed_scores": quality_scores
        }
    
    async def run_comprehensive_evaluation(self, field: str, max_papers: int = 30) -> EvaluationMetrics:
        """Run complete evaluation suite"""
        logger.info(f"Starting comprehensive evaluation for {field}")
        
        # Create ground truth if not exists
        if field not in self.ground_truth_data:
            self.create_ground_truth_dataset(field)
        
        ground_truth = self.ground_truth_data[field]
        
        # Run pipeline
        start_time = datetime.now()
        
        pipeline = ScientificPaperPipeline(
            output_dir=self.output_dir / f"eval_{field.replace(' ', '_')}",
            model_name="llama3.2:3b"
        )
        
        results = await pipeline.run_pipeline(field, max_papers)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Load papers for detailed evaluation
        papers_file = pipeline.output_dir / "papers.pkl"
        if papers_file.exists():
            import pickle
            with open(papers_file, 'rb') as f:
                papers = pickle.load(f)
        else:
            papers = []
        
        # Get cluster labels
        cluster_labels = [p.get('cluster', -1) for p in results.get('papers', [])]
        
        # Run evaluations
        content_eval = self.evaluate_content_extraction(papers, ground_truth)
        clustering_eval = self.evaluate_clustering_quality(papers, cluster_labels, ground_truth)
        coverage_eval = self.evaluate_field_coverage(papers, field)
        baseline_eval = self.benchmark_against_baselines(papers, cluster_labels)
        llm_eval = self.evaluate_llm_quality(papers)
        
        # Create comprehensive metrics
        metrics = EvaluationMetrics(
            question_extraction_rate=content_eval["question_extraction_rate"],
            method_extraction_rate=content_eval["method_extraction_rate"],
            question_coherence_score=content_eval["question_coherence_score"],
            method_coherence_score=content_eval["method_coherence_score"],
            silhouette_score=clustering_eval.get("silhouette_score", 0),
            cluster_cohesion_score=clustering_eval.get("cluster_cohesion_score", 0),
            cluster_separation_score=clustering_eval.get("cluster_separation_score", 0),
            noise_ratio=clustering_eval.get("noise_ratio", 1),
            field_coverage_score=coverage_eval["field_coverage_score"],
            temporal_coverage_score=coverage_eval["temporal_coverage_score"],
            venue_diversity_score=coverage_eval["venue_diversity_score"],
            processing_time_per_paper=processing_time / max(len(papers), 1),
            memory_efficiency_score=self._calculate_memory_efficiency(),
            api_success_rate=len(papers) / max_papers if max_papers > 0 else 0
        )
        
        # Generate evaluation report
        self.generate_evaluation_report(field, metrics, {
            "content": content_eval,
            "clustering": clustering_eval,
            "coverage": coverage_eval,
            "baselines": baseline_eval,
            "llm_quality": llm_eval,
            "processing_time": processing_time
        })
        
        return metrics
    
    def generate_evaluation_report(self, field: str, metrics: EvaluationMetrics,
                                 detailed_results: Dict):
        """Generate comprehensive evaluation report"""
        logger.info("Generating evaluation report")
        
        report = {
            "field": field,
            "evaluation_date": datetime.now().isoformat(),
            "summary_metrics": {
                "content_quality": (metrics.question_extraction_rate + metrics.method_extraction_rate) / 2,
                "clustering_quality": metrics.silhouette_score,
                "field_coverage": metrics.field_coverage_score,
                "overall_efficiency": 1 / max(metrics.processing_time_per_paper, 1)
            },
            "detailed_metrics": {
                "question_extraction_rate": metrics.question_extraction_rate,
                "method_extraction_rate": metrics.method_extraction_rate,
                "question_coherence_score": metrics.question_coherence_score,
                "method_coherence_score": metrics.method_coherence_score,
                "silhouette_score": metrics.silhouette_score,
                "cluster_cohesion_score": metrics.cluster_cohesion_score,
                "cluster_separation_score": metrics.cluster_separation_score,
                "noise_ratio": metrics.noise_ratio,
                "field_coverage_score": metrics.field_coverage_score,
                "temporal_coverage_score": metrics.temporal_coverage_score,
                "venue_diversity_score": metrics.venue_diversity_score,
                "processing_time_per_paper": metrics.processing_time_per_paper,
                "memory_efficiency_score": metrics.memory_efficiency_score,
                "api_success_rate": metrics.api_success_rate
            },
            "baseline_comparisons": detailed_results.get("baselines", {}),
            "llm_quality_analysis": detailed_results.get("llm_quality", {}),
            "recommendations": self._generate_recommendations(metrics)
        }
        
        # Save report
        with open(self.output_dir / f"evaluation_report_{field.replace(' ', '_')}.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create visualization
        self._create_evaluation_visualization(metrics, field)
        
        return report
    
    def _generate_recommendations(self, metrics: EvaluationMetrics) -> List[str]:
        """Generate actionable recommendations based on evaluation"""
        recommendations = []
        
        if metrics.question_extraction_rate < 0.7:
            recommendations.append("Improve question extraction prompts - consider more specific instructions")
        
        if metrics.method_extraction_rate < 0.6:
            recommendations.append("Enhance method extraction - may need domain-specific keywords")
        
        if metrics.silhouette_score < 0.3:
            recommendations.append("Clustering quality is low - consider different embedding models or clustering parameters")
        
        if metrics.noise_ratio > 0.3:
            recommendations.append("High noise ratio - papers may be too diverse or clustering too strict")
        
        if metrics.field_coverage_score < 0.5:
            recommendations.append("Expand search terms or data sources to improve field coverage")
        
        if metrics.processing_time_per_paper > 30:
            recommendations.append("Consider optimizing processing pipeline or using smaller models for speed")
        
        return recommendations
    
    # Helper methods
    def _calculate_semantic_overlap(self, extracted_items: List[str],
                                   ground_truth_items: List[str]) -> float:
        """Calculate semantic overlap using simple keyword matching"""
        if not extracted_items or not ground_truth_items:
            return 0.0
        
        # Simple keyword-based overlap (could be improved with embeddings)
        extracted_words = set()
        for item in extracted_items:
            extracted_words.update(item.lower().split())
        
        ground_truth_words = set()
        for item in ground_truth_items:
            ground_truth_words.update(item.lower().split())
        
        overlap = len(extracted_words.intersection(ground_truth_words))
        total = len(extracted_words.union(ground_truth_words))
        
        return overlap / total if total > 0 else 0.0
    
    def _extract_field_keywords(self, field: str) -> List[str]:
        """Extract expected keywords for a field"""
        field_keywords = {
            "physics inspired machine learning": [
                "quantum", "neural", "physics", "hamiltonian", "variational",
                "symmetry", "conservation", "dynamics", "optimization"
            ],
            "quantum machine learning": [
                "quantum", "qubit", "entanglement", "superposition", "variational",
                "quantum circuits", "quantum algorithms"
            ]
        }
        
        return field_keywords.get(field.lower(), field.split())
    
    def _calculate_keyword_coverage(self, texts: List[str], keywords: List[str]) -> float:
        """Calculate how well texts cover expected keywords"""
        if not texts or not keywords:
            return 0.0
        
        all_text = " ".join(texts).lower()
        covered_keywords = sum(1 for keyword in keywords if keyword.lower() in all_text)
        
        return covered_keywords / len(keywords)
    
    def _calculate_memory_efficiency(self) -> float:
        """Estimate memory efficiency (simplified)"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return min(memory.available / memory.total, 1.0)
        except:
            return 0.5  # Default neutral score
    
    def _create_evaluation_visualization(self, metrics: EvaluationMetrics, field: str):
        """Create evaluation visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Pipeline Evaluation: {field}', fontsize=16, fontweight='bold')
        
        # Content Quality
        axes[0,0].bar(['Questions', 'Methods'],
                     [metrics.question_extraction_rate, metrics.method_extraction_rate],
                     color=['skyblue', 'lightgreen'])
        axes[0,0].set_title('Content Extraction Rates')
        axes[0,0].set_ylabel('Extraction Rate')
        axes[0,0].set_ylim(0, 1)
        
        # Clustering Quality
        clustering_metrics = [
            metrics.silhouette_score,
            metrics.cluster_cohesion_score,
            metrics.cluster_separation_score,
            1 - metrics.noise_ratio
        ]
        axes[0,1].bar(['Silhouette', 'Cohesion', 'Separation', '1-Noise'],
                     clustering_metrics, color=['coral', 'gold', 'lightcoral', 'plum'])
        axes[0,1].set_title('Clustering Quality')
        axes[0,1].set_ylabel('Score')
        axes[0,1].set_ylim(0, 1)
        
        # Coverage Metrics
        axes[1,0].bar(['Field', 'Temporal', 'Venue'],
                     [metrics.field_coverage_score,
                      metrics.temporal_coverage_score,
                      metrics.venue_diversity_score],
                     color=['lightblue', 'lightcyan', 'lightsteelblue'])
        axes[1,0].set_title('Coverage Analysis')
        axes[1,0].set_ylabel('Coverage Score')
        axes[1,0].set_ylim(0, 1)
        
        # Efficiency
        efficiency_data = [
            min(metrics.processing_time_per_paper / 30, 1),  # Normalize to 30s
            metrics.memory_efficiency_score,
            metrics.api_success_rate
        ]
        axes[1,1].bar(['Processing Time', 'Memory Efficiency', 'API Success'],
                     efficiency_data, color=['wheat', 'khaki', 'lightgreen'])
        axes[1,1].set_title('System Efficiency')
        axes[1,1].set_ylabel('Efficiency Score')
        axes[1,1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"evaluation_{field.replace(' ', '_')}.png",
                   dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
async def run_evaluation_example():
    """Example evaluation run"""
    evaluator = PipelineEvaluator("evaluation_output")
    
    # Run comprehensive evaluation
    metrics = await evaluator.run_comprehensive_evaluation(
        "physics inspired machine learning",
        max_papers=20
    )
    
    print("Evaluation completed!")
    print(f"Overall content quality: {(metrics.question_extraction_rate + metrics.method_extraction_rate) / 2:.3f}")
    print(f"Clustering quality (silhouette): {metrics.silhouette_score:.3f}")
    print(f"Field coverage: {metrics.field_coverage_score:.3f}")
    
    return metrics

if __name__ == "__main__":
    asyncio.run(run_evaluation_example())
