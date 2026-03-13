import yaml
from pathlib import Path
from src.utils.logging import log

def load_yaml_config(config_path: str | Path) -> dict:
    """
    Load a YAML configuration file.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        log.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        log.info(f"Loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        log.error(f"Error parsing YAML file {config_path}: {e}")
        raise
