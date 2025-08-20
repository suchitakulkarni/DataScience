# Scientific Paper Analysis Pipeline

> **Note**: This project is currently work in progress. The large monolith code needs to be refactored. The code has been tested for quick analysis option.

An automated pipeline for analyzing scientific literature using multi-agent LLM orchestration and semantic clustering techniques.

## Overview

This system automatically collects, analyzes, and clusters scientific papers from multiple sources (arXiv, INSPIRE-HEP) to identify research trends, extract central questions, and categorize methodological approaches in scientific fields.

## Key Features

- **Multi-source paper collection** from arXiv and INSPIRE-HEP APIs
- **LLM-powered analysis** using Ollama for research question extraction
- **Semantic clustering** with sentence transformers and UMAP/HDBSCAN
- **Automated trend analysis** and methodological categorization
- **Interactive visualizations** of research clusters
- **Memory-optimized** for 16GB RAM systems

## Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) running locally
- Required Python packages (see installation)

## Quick Start

1. **Install Ollama and required model:**
   ```bash
   brew install ollama
   ollama serve
   ollama pull llama3.2:3b
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run analysis:**
   ```bash
   python scientific_paper_pipeline.py
   ```

## Usage Example

```python
from scientific_paper_pipeline import ScientificPaperPipeline

# Initialize pipeline
pipeline = ScientificPaperPipeline(
    output_dir="analysis_results",
    model_name="llama3.2:3b"
)

# Run analysis on a field
results = await pipeline.run_pipeline(
    field_query="physics inspired machine learning",
    max_papers=30
)
```

## Output

The pipeline generates:
- **JSON results** with research trends and cluster summaries
- **Cluster visualization** showing paper relationships
- **Detailed analysis** of research questions and methodologies
- **Pickled data** for further analysis

## System Requirements

- **RAM**: 16GB recommended for optimal performance
- **Storage**: ~1GB for results and embeddings
- **Network**: Stable internet for API calls

## Current Limitations

- Single monolithic file structure
- Limited error handling for API failures
- No configuration management
- Basic evaluation metrics

## Roadmap

- [ ] Refactor into modular components
- [ ] Add configuration files
- [ ] Implement comprehensive error handling
- [ ] Add evaluation and quality metrics
- [ ] Create web interface
- [ ] Support additional data sources

## License

MIT License
