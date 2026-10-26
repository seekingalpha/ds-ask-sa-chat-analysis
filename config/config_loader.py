import logging

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """
    Loads a YAML configuration file.

    Parameters:
    config_path (str): Path to the YAML config file.

    Returns:
    dict: Configuration dictionary.
    """
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise
