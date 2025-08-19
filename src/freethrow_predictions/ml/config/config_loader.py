from pathlib import Path
from omegaconf import OmegaConf
from .config_models import AppConfig

def load_config(config_path: Path) -> AppConfig:
    """
    Load the configuration from a YAML file using OmegaConf and convert it into a typed AppConfig object.
    
    Args:
        config_path (Path): Path to the YAML configuration file.
    
    Returns:
        AppConfig: A validated configuration instance.
    """
    try:
        # Load the YAML configuration with OmegaConf
        cfg = OmegaConf.load(config_path)
        # Convert the DictConfig to a regular dict (resolving variables, if any)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        # Use the Pydantic model for validation and type safety.
        app_config = AppConfig(**cfg_dict)
        print(f"[Config Loader] ✅ Successfully loaded configuration from {config_path}")
        return app_config
    except Exception as e:
        print(f"[Config Loader] ❌ Failed to load configuration: {e}")
        raise
