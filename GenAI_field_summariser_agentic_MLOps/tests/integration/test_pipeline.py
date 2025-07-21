import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.pipeline.main_pipeline import ScientificPaperPipeline

@pytest.mark.asyncio
async def test_full_pipeline_integration(temp_output_dir, sample_papers):
    """Test complete pipeline integration"""
    
    pipeline = ScientificPaperPipeline(
        output_dir=str(temp_output_dir),
        model_name="llama3.2:1b"  # Smaller model for testing
    )
    
    # Mock external services
    with patch.object(pipeline.collector, 'collect_arxiv_papers', return_value=sample_papers[:1]), \
         patch.object(pipeline.collector, 'collect_inspire_papers', return_value=sample_papers[1:]), \
         patch('ollama.generate') as mock_ollama:
        
        mock_ollama.return_value = {
            'response': '- Test question 1?\n- Test question 2?'
        }
        
        results = await pipeline.run_pipeline("test query", max_papers=2)
        
        assert results['query'] == "test query"
        assert results['trends']['total_papers'] >= 1
        assert 'cluster_summaries' in results
        
        # Check output files were created
        assert (temp_output_dir / "results.json").exists()
        assert (temp_output_dir / "papers.pkl").exists()

@pytest.mark.integration
def test_pipeline_with_real_ollama():
    """Test pipeline with real Ollama service (requires Ollama running)"""
    # This test should be skipped if Ollama is not available
    try:
        import ollama
        ollama.list()
    except:
        pytest.skip("Ollama not available")
    
    # Test with real Ollama...
    pass
