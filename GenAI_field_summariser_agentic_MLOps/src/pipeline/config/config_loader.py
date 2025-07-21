#!/usr/bin/env python3
"""
Configuration loading and validation system
src/pipeline/config/config_loader.py
"""

import yaml
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, validator, Field
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class CollectionConfig(BaseModel):
    """Configuration for paper collection"""
    rate_limit: float = 1.0
    max_papers_per_source: int = 100
    days_back: int = 90
    sources: list = ["arxiv", "inspire"]
    api_keys: Dict[str, str] = Field(default_factory=dict)
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    @validator('rate_limit')
    def rate_limit_positive(cls, v):
        if v < 0:
            raise ValueError('rate_limit must be non-negative')
        return v
    
    @validator('retry_attempts')
    def retry_attempts_positive(cls, v):
        if v < 0:
            raise ValueError('retry_attempts must be non-negative')
        return v


class OllamaConfig(BaseModel):
    """Ollama model configuration"""
    model_name: str = "llama3.2:3b"
    temperature: float = 0.3
    num_predict: int = 200
    timeout: int = 30
    base_url: str = "http://localhost:11434"
    max_retries: int = 3
    
    @validator('temperature')
    def temperature_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('temperature must be between 0 and 1')
        return v
    
    @validator('timeout')
    def timeout_positive(cls, v):
        if v <= 0:
            raise ValueError('timeout must be positive')
        return v


class EmbeddingConfig(BaseModel):
    """Embedding model configuration"""
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize_embeddings: bool = True
    cache_embeddings: bool = True
    device: str = "auto"  # auto, cpu, cuda
    max_sequence_length: int = 512
    
    @validator('batch_size')
    def batch_size_positive(cls, v):
        if v <= 0:
            raise ValueError('batch_size must be positive')
        return v
    
    @validator('max_sequence_length')
    def max_length_positive(cls, v):
        if v <= 0:
            raise ValueError('max_sequence_length must be positive')
        return v


class ClusteringConfig(BaseModel):
    """Clustering configuration"""
    n_neighbors: int = 15
    min_cluster_size: int = 3
    umap_components: int = 2
    metric: str = "cosine"
    random_state: int = 42
    cluster_selection_epsilon: float = 0.0
    
    @validator('n_neighbors', 'min_cluster_size', 'umap_components')
    def positive_integers(cls, v):
        if v <= 0:
            raise ValueError('Must be positive integer')
        return v
    
    @validator('metric')
    def valid_metric(cls, v):
        valid_metrics = ['cosine', 'euclidean', 'manhattan', 'hamming']
        if v not in valid_metrics:
            raise ValueError(f'metric must be one of {valid_metrics}')
        return v


class AnalysisConfig(BaseModel):
    """Analysis configuration"""
    clustering: ClusteringConfig = ClusteringConfig()
    max_features: int = 100
    stop_words: str = "english"
    min_question_length: int = 10
    max_questions_per_paper: int = 3
    max_methods_per_paper: int = 5
    enable_sentiment_analysis: bool = False
    min_confidence_score: float = 0.5
    
    @validator('min_confidence_score')
    def confidence_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('min_confidence_score must be between 0 and 1')
        return v


class OutputConfig(BaseModel):
    """Output configuration"""
    base_dir: str = "output"
    save_embeddings: bool = True
    save_plots: bool = True
    plot_dpi: int = 300
    create_subdirs: bool = True
    export_formats: list = ["json", "csv"]
    timestamp_dirs: bool = True
    
    @validator('plot_dpi')
    def dpi_positive(cls, v):
        if v <= 0:
            raise ValueError('plot_dpi must be positive')
        return v
    
    @validator('export_formats')
    def valid_formats(cls, v):
        valid_formats = ['json', 'csv', 'xlsx', 'parquet']
        for fmt in v:
            if fmt not in valid_formats:
                raise ValueError(f'export format {fmt} not in {valid_formats}')
        return v


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_file_size: int = 10_000_000  # 10MB
    backup_count: int = 3
    console_output: bool = True
    
    @validator('level')
    def valid_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log level must be one of {valid_levels}')
        return v.upper()


class DatabaseConfig(BaseModel):
    """Database configuration for caching and storage"""
    enabled: bool = False
    type: str = "sqlite"  # sqlite, postgresql, mysql
    host: str = "localhost"
    port: int = 5432
    database: str = "pipeline_cache"
    username: Optional[str] = None
    password: Optional[str] = None
    connection_pool_size: int = 5
    
    @validator('type')
    def valid_db_type(cls, v):
        valid_types = ['sqlite', 'postgresql', 'mysql']
        if v not in valid_types:
            raise ValueError(f'database type must be one of {valid_types}')
        return v


class ResourceConfig(BaseModel):
    """Resource limits configuration"""
    max_memory_gb: int = 8
    max_concurrent_requests: int = 5
    embedding_cache_size: int = 1000
    max_workers: int = 4
    gpu_memory_fraction: float = 0.8
    
    @validator('gpu_memory_fraction')
    def gpu_memory_range(cls, v):
        if not 0 < v <= 1:
            raise ValueError('gpu_memory_fraction must be between 0 and 1')
        return v


class MonitoringConfig(BaseModel):
    """Monitoring and metrics configuration"""
    enabled: bool = False
    metrics_port: int = 9090
    health_check_interval: int = 30
    alert_thresholds: Dict[str, float] = Field(default_factory=lambda: {
        'memory_usage': 0.9,
        'error_rate': 0.1,
        'response_time': 30.0
    })


class PipelineConfig(BaseModel):
    """Main pipeline configuration"""
    collection: CollectionConfig = CollectionConfig()
    ollama: OllamaConfig = OllamaConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    output: OutputConfig = OutputConfig()
    logging: LoggingConfig = LoggingConfig()
    database: DatabaseConfig = DatabaseConfig()
    resources: ResourceConfig = ResourceConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    
    # Pipeline metadata
    version: str = "1.0.0"
    description: str = "Research paper analysis pipeline"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return self.dict()
    
    def to_json(self, indent: int = 2) -> str:
        """Convert config to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save_to_file(self, file_path: Union[str, Path]):
        """Save configuration to file"""
        path = Path(file_path)
        
        if path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                f.write(self.to_json())
        else:  # Default to YAML
            with open(path, 'w') as f:
                yaml.safe_dump(self.to_dict(), f, indent=2)


class ConfigError(Exception):
    """Configuration related errors"""
    pass


class ConfigValidator:
    """Additional configuration validation logic"""
    
    @staticmethod
    def validate_config(config: PipelineConfig) -> list:
        """Validate configuration and return warnings"""
        warnings = []
        
        # Check resource constraints
        if config.embedding.batch_size > 64 and config.resources.max_memory_gb < 4:
            warnings.append("Large embedding batch size with limited memory may cause issues")
        
        # Check clustering parameters
        if config.analysis.clustering.min_cluster_size > config.analysis.clustering.n_neighbors:
            warnings.append("min_cluster_size should not exceed n_neighbors")
        
        # Check output directory permissions
        try:
            output_path = Path(config.output.base_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            test_file = output_path / ".write_test"
            test_file.touch()
            test_file.unlink()
        except Exception:
            warnings.append(f"Cannot write to output directory: {config.output.base_dir}")
        
        # Check database configuration
        if config.database.enabled and config.database.type == "sqlite":
            if not config.database.database.endswith(('.db', '.sqlite', '.sqlite3')):
                warnings.append("SQLite database should have appropriate file extension")
        
        return warnings


class ConfigLoader:
    """Configuration loader with environment support"""
    
    def __init__(self, config_dir: Optional[str] = None):
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Default to config directory relative to this file
            self.config_dir = Path(__file__).parent.parent.parent.parent / "config"
        
        self.validator = ConfigValidator()
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML file with error handling"""
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {file_path}: {e}")
        except FileNotFoundError:
            raise ConfigError(f"Config file not found: {file_path}")
        except Exception as e:
            raise ConfigError(f"Error reading config file {file_path}: {e}")
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge configuration dictionaries"""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _create_default_config_files(self):
        """Create default configuration files if they don't exist"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        default_config = {
            'collection': {
                'rate_limit': 1.0,
                'max_papers_per_source': 100,
                'days_back': 90,
                'sources': ['arxiv', 'inspire']
            },
            'ollama': {
                'model_name': 'llama3.2:3b',
                'temperature': 0.3
            },
            'output': {
                'base_dir': 'output'
            },
            'logging': {
                'level': 'INFO'
            }
        }
        
        default_path = self.config_dir / "default.yaml"
        if not default_path.exists():
            with open(default_path, 'w') as f:
                yaml.safe_dump(default_config, f, indent=2)
            logger.info(f"Created default config: {default_path}")
    
    def load_config(self, environment: str = "default", config_file: Optional[str] = None, 
                   validate: bool = True, create_defaults: bool = True) -> PipelineConfig:
        """
        Load configuration for specified environment
        
        Args:
            environment: Environment name (default, development, production)
            config_file: Optional path to specific config file
            validate: Whether to run additional validation
            create_defaults: Whether to create default config files if missing
            
        Returns:
            Validated PipelineConfig instance
        """
        try:
            if create_defaults:
                self._create_default_config_files()
            
            if config_file:
                # Load specific config file
                config_path = Path(config_file)
                config_data = self._load_yaml_file(config_path)
            else:
                # Load base config
                default_path = self.config_dir / "default.yaml"
                if not default_path.exists():
                    logger.warning(f"Default config not found at {default_path}, using defaults")
                    config_data = {}
                else:
                    config_data = self._load_yaml_file(default_path)
                
                # Load environment-specific overrides
                if environment != "default":
                    env_path = self.config_dir / f"{environment}.yaml"
                    if env_path.exists():
                        env_config = self._load_yaml_file(env_path)
                        config_data = self._merge_configs(config_data, env_config)
                        logger.info(f"Loaded environment config: {environment}")
                    else:
                        logger.warning(f"Environment config not found: {env_path}")
            
            # Override with environment variables
            config_data = self._apply_env_overrides(config_data)
            
            # Create and validate config
            config = PipelineConfig(**config_data)
            
            if validate:
                warnings = self.validator.validate_config(config)
                for warning in warnings:
                    logger.warning(f"Config validation: {warning}")
            
            logger.info(f"Configuration loaded successfully for environment: {environment}")
            return config
            
        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {e}")
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        # Define environment variable mappings
        env_mappings = {
            'PIPELINE_RATE_LIMIT': ('collection', 'rate_limit'),
            'PIPELINE_MAX_PAPERS': ('collection', 'max_papers_per_source'),
            'PIPELINE_DAYS_BACK': ('collection', 'days_back'),
            'OLLAMA_MODEL': ('ollama', 'model_name'),
            'OLLAMA_TEMPERATURE': ('ollama', 'temperature'),
            'OLLAMA_BASE_URL': ('ollama', 'base_url'),
            'EMBEDDING_MODEL': ('embedding', 'model_name'),
            'EMBEDDING_BATCH_SIZE': ('embedding', 'batch_size'),
            'EMBEDDING_DEVICE': ('embedding', 'device'),
            'OUTPUT_DIR': ('output', 'base_dir'),
            'LOG_LEVEL': ('logging', 'level'),
            'LOG_FILE': ('logging', 'file'),
            'MAX_MEMORY_GB': ('resources', 'max_memory_gb'),
            'MAX_WORKERS': ('resources', 'max_workers'),
            'DB_ENABLED': ('database', 'enabled'),
            'DB_TYPE': ('database', 'type'),
            'DB_HOST': ('database', 'host'),
            'DB_PORT': ('database', 'port'),
            'DB_NAME': ('database', 'database'),
            'DB_USER': ('database', 'username'),
            'DB_PASSWORD': ('database', 'password'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Ensure section exists
                if section not in config_data:
                    config_data[section] = {}
                
                # Convert type based on existing value or reasonable defaults
                if key in ['rate_limit', 'temperature', 'gpu_memory_fraction']:
                    config_data[section][key] = float(value)
                elif key in ['max_papers_per_source', 'max_memory_gb', 'timeout', 
                           'batch_size', 'max_workers', 'days_back', 'port']:
                    config_data[section][key] = int(value)
                elif key in ['normalize_embeddings', 'save_plots', 'enabled']:
                    config_data[section][key] = value.lower() in ('true', '1', 'yes')
                else:
                    config_data[section][key] = value
                
                logger.debug(f"Applied environment override: {env_var}={value}")
        
        return config_data
    
    def list_available_configs(self) -> list:
        """List available configuration files"""
        if not self.config_dir.exists():
            return []
        
        config_files = []
        for file_path in self.config_dir.glob("*.yaml"):
            config_files.append(file_path.stem)
        
        return sorted(config_files)


# Global config loader instance
_config_loader = ConfigLoader()

def load_config(environment: str = "default", config_file: Optional[str] = None, 
               validate: bool = True) -> PipelineConfig:
    """Load configuration (convenience function)"""
    return _config_loader.load_config(environment, config_file, validate)


def setup_logging(config: PipelineConfig):
    """Setup logging based on configuration"""
    log_config = config.logging
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure logging
    formatter = logging.Formatter(log_config.format)
    
    if log_config.file:
        # Setup rotating file handler
        from logging.handlers import RotatingFileHandler
        
        # Ensure log directory exists
        log_path = Path(log_config.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_config.file,
            maxBytes=log_config.max_file_size,
            backupCount=log_config.backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    if log_config.console_output:
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Set level
    root_logger.setLevel(getattr(logging, log_config.level))
    
    logger.info(f"Logging configured: level={log_config.level}")


def get_config_schema() -> Dict[str, Any]:
    """Get the configuration schema for documentation"""
    return PipelineConfig.schema()


def validate_config_file(config_path: Union[str, Path]) -> tuple[bool, list]:
    """Validate a configuration file without loading it fully"""
    try:
        loader = ConfigLoader()
        config = loader.load_config(config_file=str(config_path), validate=True)
        return True, []
    except Exception as e:
        return False, [str(e)]


# Example usage and validation
if __name__ == "__main__":
    # Test configuration loading
    try:
        print("Testing configuration system...")
        
        # Test default config
        config = load_config("development")
        print("✅ Configuration loaded successfully")
        print(f"Ollama model: {config.ollama.model_name}")
        print(f"Max papers: {config.collection.max_papers_per_source}")
        print(f"Output directory: {config.output.base_dir}")
        
        # Setup logging
        setup_logging(config)
        logger.info("Configuration system test completed successfully")
        
        # Test configuration validation
        warnings = ConfigValidator.validate_config(config)
        if warnings:
            print(f"⚠️  Configuration warnings: {len(warnings)}")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("✅ No configuration warnings")
        
        # Test configuration export
        print("\n📄 Configuration export test:")
        print(f"JSON export preview:\n{config.to_json()[:200]}...")
        
        # Test available configs
        loader = ConfigLoader()
        available_configs = loader.list_available_configs()
        print(f"\n📋 Available configurations: {available_configs}")
        
        # Test environment variable override
        print(f"\n🔧 Current settings:")
        print(f"  - Rate limit: {config.collection.rate_limit}")
        print(f"  - Batch size: {config.embedding.batch_size}")
        print(f"  - Temperature: {config.ollama.temperature}")
        print(f"  - Max memory: {config.resources.max_memory_gb}GB")
        
        # Test schema generation
        schema = get_config_schema()
        print(f"\n📊 Configuration schema has {len(schema.get('properties', {}))} top-level properties")
        
        print("\n✨ All configuration tests passed!")
        
    except ConfigError as e:
        print(f"❌ Configuration error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    # Demonstrate configuration usage patterns
    print("\n" + "="*50)
    print("CONFIGURATION USAGE EXAMPLES")
    print("="*50)
    
    # Example 1: Loading different environments
    print("\n1. Loading different environments:")
    environments = ["default", "development", "production"]
    for env in environments:
        try:
            cfg = load_config(env, validate=False)
            print(f"   {env}: ✅ (Ollama: {cfg.ollama.model_name})")
        except Exception as e:
            print(f"   {env}: ❌ ({str(e)[:50]}...)")
    
    # Example 2: Environment variable overrides
    print("\n2. Environment variable override examples:")
    print("   export PIPELINE_RATE_LIMIT=2.0")
    print("   export OLLAMA_MODEL=llama3.2:7b")
    print("   export LOG_LEVEL=DEBUG")
    print("   export OUTPUT_DIR=/custom/output")
    
    # Example 3: Custom configuration
    print("\n3. Creating custom configuration:")
    custom_config = PipelineConfig(
        collection=CollectionConfig(
            rate_limit=0.5,
            max_papers_per_source=50,
            sources=["arxiv"]
        ),
        ollama=OllamaConfig(
            model_name="llama3.2:1b",
            temperature=0.1
        )
    )
    print(f"   Custom config created with {custom_config.collection.max_papers_per_source} max papers")
    
    # Example 4: Configuration validation
    print("\n4. Configuration validation:")
    is_valid, errors = validate_config_file("config/default.yaml")
    if is_valid:
        print("   Default config file: ✅ Valid")
    else:
        print(f"   Default config file: ❌ Errors: {errors}")
    
    print("\n" + "="*50)
