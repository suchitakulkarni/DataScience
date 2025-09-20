import yaml
from pathlib import Path
from typing import Optional
from .config_schema import PipelineConfig

class ConfigError(Exception):
    pass

def load_config(environment: str = "default", config_file: Optional[str] = None) -> PipelineConfig:
    """Load and validate configuration"""
    
    if config_file:
        config_path = Path(config_file)
    else:
        config_dir = Path(__file__).parent.parent.parent.parent / "config"
        config_path = config_dir / f"{environment}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        
        # Load default first if not loading default
        if environment != "default":
            default_path = config_path.parent / "default.yaml"
            if default_path.exists():
                with open(default_path) as f:
                    default_data = yaml.safe_load(f)
                # Merge configs (environment overrides default)
                config_data = {**default_data, **config_data}
        
        return PipelineConfig(**config_data)
        
    except Exception as e:
        raise ConfigError(f"Failed to load/validate config: {e}")
