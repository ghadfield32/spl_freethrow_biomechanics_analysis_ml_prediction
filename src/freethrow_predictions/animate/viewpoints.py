import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def update_3d_view(ax: plt.Axes, elev: float, azim: float, data_zoom: float, zlim: float) -> None:
    """
    Update the view of a 3D axis independently.
    
    This function sets the camera view (elevation and azimuth) and adjusts the axis
    limits using the provided data_zoom factor and z-axis limit.
    
    Parameters:
      - ax (plt.Axes): The 3D axis to update.
      - elev (float): The elevation angle.
      - azim (float): The azimuth angle.
      - data_zoom (float): The zoom factor to adjust x, y, and z limits.
      - zlim (float): The maximum value for the z-axis.
    
    Returns:
      - None
    """
    # Set the camera view
    ax.view_init(elev=elev, azim=azim)
    
    # Retrieve current x and y limits and adjust them based on the zoom factor.
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_xlim([x_min / data_zoom, x_max / data_zoom])
    ax.set_ylim([y_min / data_zoom, y_max / data_zoom])
    # Set the z-axis limit from 0 to zlim adjusted by the zoom factor.
    ax.set_zlim([0, zlim / data_zoom])
    
    logger.debug(f"3D view updated: elev={elev}, azim={azim}, data_zoom={data_zoom}, zlim={zlim}")
