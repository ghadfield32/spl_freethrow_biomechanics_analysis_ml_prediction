import pandas as pd
import logging
import logging.config
import numpy as np
import ast
from pathlib import Path
from typing import Any, Dict, Optional
from ml.config.config_models import AppConfig  # Adjust the import based on your project structure
from ml.config.config_loader import load_config  # Assuming you have a config_loader module

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load dataset from the specified path.
    
    :param path: Path to the CSV file.
    :return: Loaded DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def setup_logging(config: AppConfig, log_file_path: Path) -> logging.Logger:
    """
    Set up logging based on the configuration.
    
    :param config: Application configuration.
    :param log_file_path: Path to the log file.
    :return: Configured logger.
    """
    log_level = config.logging.level.upper()
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'standard',
                'stream': 'ext://sys.stdout',
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': log_level,
                'formatter': 'standard',
                'filename': str(log_file_path),
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file'],
                'level': log_level,
                'propagate': True
            },
        }
    }
    logging.config.dictConfig(logging_config)
    return logging.getLogger(__name__)


def load_configuration(config_path: Path) -> AppConfig:
    """
    Load the application configuration from a YAML file.

    :param config_path: Path to the configuration YAML file.
    :return: AppConfig object containing configuration parameters.
    """
    try:
        config: AppConfig = load_config(config_path)
        logging.getLogger(__name__).info(f"Configuration loaded successfully from {config_path}.")
        return config
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load configuration from {config_path}: {e}")
        raise

def initialize_logger(config: AppConfig, log_file: Path) -> logging.Logger:
    """
    Initialize and return a logger based on the configuration.

    :param config: AppConfig object containing configuration parameters.
    :param log_file: Path to the log file.
    :return: Configured logger instance.
    """
    try:
        logger = setup_logging(config, log_file)
        logger.info("Logger initialized successfully.")
        return logger
    except Exception as e:
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger(__name__).error(f"Failed to initialize logger: {e}")
        raise


if __name__ == "__main__":
    # Test code to verify the functions in shap_utils.py
    print("Testing shap_utils.py module...")

    # Step 1: Load Configuration
    config_path = Path('../../data/model/preprocessor_config/preprocessor_config.yaml')
    try:
        config: AppConfig = load_config(config_path)
        print(f"✅ Configuration loaded successfully from {config_path}.")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        exit(1)

    # Step 2: Set Up Logging
    log_file = Path(config.paths.log_file).resolve()
    try:
        logger = setup_logging(config, log_file)
        logger.info("Logging has been set up successfully.")
    except Exception as e:
        print(f"❌ Failed to set up logging: {e}")
        exit(1)

    # Step 3: Load Dataset
    raw_data_path = Path(config.paths.data_dir).resolve() / config.paths.raw_data
    try:
        df = load_dataset(raw_data_path)
        print(f"✅ Dataset loaded successfully from {raw_data_path}.")
        print(f"📊 Dataset Columns: {df.columns.tolist()}")
        logger.info(f"Dataset loaded with shape: {df.shape}")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        exit(1)


