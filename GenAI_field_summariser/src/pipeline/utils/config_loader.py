# src/pipeline/utils/config_loader.py
"""
Configuration loading and management
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from ..models.schemas import PipelineConfig


class ConfigLoader:
    """Loads and manages pipeline configuration"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._cache: Dict[str, Any] = {}
    
    def load_config(self, 
                   environment: str = "default",
                   config_overrides: Optional[Dict[str, Any]] = None) -> PipelineConfig:
        """Load configuration for specified environment"""
        
        # Load base configuration
        base_config = self._load_yaml_file("default.yaml")
        
        # Load environment-specific config if different from default
        if environment != "default":
            env_config_file = f"{environment}.yaml"
            if (self.config_dir / env_config_file).exists():
                env_config = self._load_yaml_file(env_config_file)
                base_config = self._merge_configs(base_config, env_config)
        
        # Apply environment variable overrides
        base_config = self._apply_env_overrides(base_config)
        
        # Apply manual overrides
        if config_overrides:
            base_config = self._merge_configs(base_config, config_overrides)
        
        # Validate and return
        return PipelineConfig(**base_config)
    
    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Check cache first
        cache_key = str(file_path)
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        self._cache[cache_key] = config
        return config.copy()
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        env_mappings = {
            'PIPELINE_OLLAMA_MODEL': 'analyzer.ollama_model',
            'PIPELINE_MAX_PAPERS': 'collector.max_results_per_source',
            'PIPELINE_OUTPUT_DIR': 'output_dir',
            'PIPELINE_LOG_LEVEL': 'log_level',
            'PIPELINE_BATCH_SIZE': 'analyzer.batch_size'
        }
        
        result = config.copy()
        
        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                self._set_nested_value(result, config_path, os.environ[env_var])
        
        return result
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: str):
        """Set nested configuration value using dot notation"""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Type conversion for common cases
        final_key = keys[-1]
        if value.lower() in ('true', 'false'):
            current[final_key] = value.lower() == 'true'
        elif value.isdigit():
            current[final_key] = int(value)
        elif value.replace('.', '').isdigit():
            current[final_key] = float(value)
        else:
            current[final_key] = value
