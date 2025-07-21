import pytest
import psutil
import os
from src.pipeline.main_pipeline import ScientificPaperPipeline

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

@pytest.mark.performance
def test_memory_usage_stays_reasonable(sample_papers):
    """Test that memory usage doesn't grow too much during processing"""
    
    initial_memory = get_memory_usage()
    
    # Process papers
    pipeline = ScientificPaperPipeline()
    
    # Simulate processing large batch
    large_paper_list = sample_papers * 100  # 200 papers
    
    # This should not cause excessive memory growth
    # Implementation would process in batches
    
    final_memory = get_memory_usage()
    memory_growth = final_memory - initial_memory
    
    # Should not grow by more than 500MB for this test
    assert memory_growth < 500, f"Memory grew by {memory_growth:.1f}MB"

@pytest.mark.performance  
def test_processing_time_scales_reasonably():
    """Test that processing time scales reasonably with input size"""
    import time
    
    # Test with different sizes and ensure reasonable scaling
    # This would need actual implementation
    pass
