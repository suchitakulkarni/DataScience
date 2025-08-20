#!/usr/bin/env python3
"""
Simple Example: Scientific Paper Analysis
Minimal version for quick testing and demonstration
"""

import asyncio
import json
from datetime import datetime
import numpy as np
from pathlib import Path

# Import the main pipeline
from scientific_paper_pipeline import ScientificPaperPipeline

async def quick_analysis_example():
    """
    Quick analysis example for testing
    Analyzes 15 papers to demonstrate all features quickly
    """
    print(" Starting Quick Scientific Paper Analysis")
    print("=" * 60)
    
    # Initialize pipeline with memory-optimized settings
    pipeline = ScientificPaperPipeline(
        output_dir="quick_test_output",
        model_name="llama3.2:3b"
    )
    
    # Run analysis on a manageable dataset
    field_query = "physics inspired machine learning"
    print(f" Analyzing field: {field_query}")
    print(f" Target papers: 15 (for quick demonstration)")
    
    try:
        # Run the pipeline
        results = await pipeline.run_pipeline(
            field_query=field_query,
            max_papers=5  # Small number for quick testing
        )
        
        # Print comprehensive summary
        pipeline.print_summary(results)
        
        # Additional insights
        print_detailed_insights(results)
        
        return results
        
    except Exception as e:
        print(f" Error during analysis: {e}")
        print(" Try reducing max_papers or check if Ollama is running")
        return None

def print_detailed_insights(results):
    """Print additional detailed insights"""
    print("\n" + " DETAILED INSIGHTS")
    print("-" * 40)
    
    papers = results.get('papers', [])
    clusters = results.get('cluster_summaries', {})
    
    # Question analysis
    all_questions = []
    for paper in papers:
        if paper.get('central_questions'):
            all_questions.extend(paper['central_questions'])
    
    if all_questions:
        print(f"\n MOST COMMON RESEARCH QUESTIONS:")
        question_freq = {}
        for q in all_questions:
            # Simple keyword extraction
            keywords = [word.lower() for word in q.split()
                       if len(word) > 4 and word.isalpha()]
            for keyword in keywords:
                question_freq[keyword] = question_freq.get(keyword, 0) + 1
        
        top_keywords = sorted(question_freq.items(), key=lambda x: x[1], reverse=True)[:8]
        for keyword, freq in top_keywords:
            print(f"    {keyword.capitalize()}: mentioned {freq} times")
    
    # Method analysis
    all_methods = []
    for paper in papers:
        if paper.get('methods'):
            all_methods.extend(paper['methods'])
    
    if all_methods:
        print(f"\n  METHODOLOGICAL LANDSCAPE:")
        method_freq = {}
        for method in all_methods:
            method_lower = method.lower()
            method_freq[method_lower] = method_freq.get(method_lower, 0) + 1
        
        top_methods = sorted(method_freq.items(), key=lambda x: x[1], reverse=True)[:6]
        for method, freq in top_methods:
            print(f"   {method.title()}: used in {freq} papers")
    
    # Cluster insights
    if clusters:
        print(f"\n RESEARCH COMMUNITY STRUCTURE:")
        total_clustered = sum(cluster['size'] for cluster in clusters.values())
        total_papers = len(papers)
        noise_papers = total_papers - total_clustered
        
        print(f"    {len(clusters)} distinct research clusters identified")
        print(f"    {total_clustered}/{total_papers} papers successfully clustered")
        if noise_papers > 0:
            print(f"    {noise_papers} papers represent novel/outlier approaches")
        
        # Find most coherent cluster
        if clusters:
            largest_cluster = max(clusters.values(), key=lambda x: x['size'])
            cluster_name = [name for name, data in clusters.items() if data == largest_cluster][0]
            print(f"    Main research focus: {cluster_name} ({largest_cluster['size']} papers)")

async def comparative_example():
    """
    Example showing how to compare multiple subfields
    """
    print("\n" + " COMPARATIVE ANALYSIS EXAMPLE")
    print("=" * 60)
    
    subfields = [
        "quantum machine learning",
        "physics informed neural networks"
    ]
    
    comparison_results = {}
    
    for field in subfields:
        print(f"\n Analyzing: {field}")
        
        pipeline = ScientificPaperPipeline(
            output_dir=f"comparison_{field.replace(' ', '_')}",
            model_name="llama3.2:3b"
        )
        
        results = await pipeline.run_pipeline(field, max_papers=10)  # Small for demo
        comparison_results[field] = results
        
        # Brief summary
        trends = results['trends']
        print(f"    Analyzed {trends['total_papers']} papers")
        print(f"    Top themes: {', '.join(trends['common_question_themes'][:3])}")
    
    # Cross-field comparison
    print(f"\n CROSS-FIELD INSIGHTS:")
    for field, results in comparison_results.items():
        themes = results['trends']['common_question_themes'][:3]
        methods = results['trends']['common_methods'][:3]
        print(f"\n   {field.upper()}:")
        print(f"   ├─ Key themes: {', '.join(themes)}")
        print(f"   └─ Main methods: {', '.join(methods)}")

def check_system_requirements():
    """
    Check if the system is ready for analysis
    """
    print(" SYSTEM REQUIREMENTS CHECK")
    print("-" * 40)
    
    # Check Ollama
    try:
        import ollama
        models = ollama.list()
        print(" Ollama connection successful")
        
        # Check for recommended models
        available_models = [model['name'] for model in models['models']]
        recommended = ['llama3.2:3b', 'phi3.5:3.8b', 'qwen2:1.5b']
        
        found_models = [model for model in recommended if any(model in avail for avail in available_models)]
        
        if found_models:
            print(f" Found recommended models: {', '.join(found_models)}")
        else:
            print("  No recommended models found. Install with:")
            print("   ollama pull llama3.2:3b")
        
    except Exception as e:
        print(f" Ollama not accessible: {e}")
        print(" Please install and start Ollama:")
        print("   brew install ollama")
        print("   ollama serve")
        return False
    
    # Check memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        total_gb = memory.total / (1024**3)
        
        print(f" Available RAM: {available_gb:.1f}GB / {total_gb:.1f}GB")
        
        if available_gb >= 8:
            print(" Sufficient memory for analysis")
        elif available_gb >= 6:
            print("  Limited memory - recommend smaller batch sizes")
        else:
            print(" Low memory - may need optimization")
            
    except ImportError:
        print("  Install psutil for memory monitoring: pip install psutil")
    
    # Check core dependencies
    required_packages = [
        'sentence_transformers', 'umap', 'hdbscan',
        'arxiv', 'requests', 'numpy', 'pandas'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if not missing_packages:
        print(" All required packages installed")
    else:
        print(f" Missing packages: {', '.join(missing_packages)}")
        print(" Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("\n System ready for analysis!")
    return True

async def main():
    """Main example runner"""
    
    # Check system first
    if not check_system_requirements():
        print("\n Please resolve system requirements before proceeding")
        return
    
    print("\n" + "="*60)
    print("SCIENTIFIC PAPER ANALYSIS PIPELINE - EXAMPLES")
    print("="*60)
    
    # Menu system
    print("\nChoose an example to run:")
    print("1. Quick Analysis (15 papers, ~5-10 minutes)")
    print("2. Comparative Analysis (2 fields, ~10-15 minutes)")
    print("3. System Check Only")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        results = await quick_analysis_example()
        
        if results:
            print(f"\n Results saved to: quick_test_output/")
            print(f" View clusters.png for visualization")
            print(f" Full results in results.json")
    
    elif choice == "2":
        await comparative_example()
        print(f"\n Results saved to comparison directories")
    
    elif choice == "3":
        print("\n System check complete")
    
    else:
        print("Invalid choice. Running quick analysis by default...")
        await quick_analysis_example()

if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
