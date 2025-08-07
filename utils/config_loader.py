"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Optional path to config file (defaults to config/settings.yaml)
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path("config/settings.yaml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_sections = ["model", "embeddings", "vector_store", "paths", "logging", "ui"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate model config
    model_required = ["id", "dtype", "device"]
    for key in model_required:
        if key not in config["model"]:
            raise ValueError(f"Missing required model config: {key}")
    
    # Validate paths
    if "pdf_dir" not in config["paths"]:
        raise ValueError("Missing required path: pdf_dir")
    
    return True


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configurations, with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    import copy
    merged = copy.deepcopy(base_config)
    
    def deep_merge(base: Dict, override: Dict) -> Dict:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    return deep_merge(merged, override_config)