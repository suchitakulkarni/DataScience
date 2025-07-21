import pytest
import tempfile
import yaml
from pathlib import Path

from src.pipeline.config.config_loader import load_config, ConfigError

def test_load_default_config():
    """Test loading default configuration"""
    config = load_config()
    
    assert config.collection.rate_limit == 1.0
    assert config.models.ollama.model_name == "llama3.2:3b"
    assert "arxiv" in config.collection.sources

def test_load_development_config():
    """Test loading development configuration with overrides"""
    config = load_config("development")
    
    # Should have development overrides
    assert config.collection.max_papers_per_source == 20
    assert config.logging.level == "DEBUG"
    
    # Should still have default values for non-overridden fields
    assert config.collection.rate_limit == 1.0

def test_invalid_config_raises_error():
    """Test that invalid config raises ConfigError"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = Path(temp_dir) / "invalid.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({"collection": {"rate_limit": -1}}, f)
        
        with pytest.raises(ConfigError):
            load_config(config_file=str(config_file))

def test_missing_config_file():
    """Test handling of missing config file"""
    with pytest.raises(FileNotFoundError):
        load_config(config_file="nonexistent.yaml")
