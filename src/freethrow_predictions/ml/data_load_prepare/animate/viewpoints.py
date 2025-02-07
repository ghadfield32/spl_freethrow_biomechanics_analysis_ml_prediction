import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Define common viewpoints
COMMON_VIEWPOINTS = {
    "side_view_right": {"elev": 0, "azim": 90},
    "side_view_left": {"elev": 0, "azim": -90},
    "top_down": {"elev": 90, "azim": 0},
    "diagonal_view": {"elev": 45, "azim": 45},
    "player_centric": {"elev": 30, "azim": 0},
    "diagonal_player_centric": {"elev": 30, "azim": 45},
    "inverse_player_centric": {"elev": 30, "azim": 180}
}

def get_viewpoint(name: str) -> dict:
    """
    Retrieve viewpoint parameters by name.
    
    Parameters:
    - name (str): The name of the viewpoint.
    
    Returns:
    - dict: Dictionary containing 'elev' and 'azim'.
    """
    try:
        viewpoint = COMMON_VIEWPOINTS[name]
        logger.debug(f"Retrieved viewpoint '{name}': {viewpoint}")
        return viewpoint
    except KeyError:
        logger.error(f"Viewpoint '{name}' not found. Available viewpoints: {list(COMMON_VIEWPOINTS.keys())}")
        raise ValueError(f"Viewpoint '{name}' not found. Choose from {list(COMMON_VIEWPOINTS.keys())}")
