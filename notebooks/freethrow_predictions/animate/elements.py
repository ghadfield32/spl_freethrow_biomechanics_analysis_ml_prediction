import logging
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def initialize_elements(
    ax: plt.Axes,
    connections: list,
    player_color: str,
    player_lw: float,
    ball_color: str,
    ball_size: float,
    debug: bool = False
) -> (dict, plt.Line2D, plt.Text, plt.Text, plt.Text, dict):
    """
    Initialize plot elements for the player skeleton, ball, text annotations,
    and now also creates a marker (small dot) for every unique joint.
    
    Returns:
        lines (dict): Dictionary of line objects for each connection.
        ball (plt.Line2D): The ball plot object.
        release_text (plt.Text): Text object for release point indicator.
        motion_text (plt.Text): Text object for motion phase indicator.
        distance_text (plt.Text): Text object for distance to hoop.
        joint_markers (dict): Dictionary of marker objects for each unique joint.
    """
    try:
        # 1) SKELETON LINES
        lines = {connection: ax.plot([], [], [], c=player_color, lw=player_lw)[0] for connection in connections}
        
        # 2) BALL MARKER
        ball, = ax.plot([], [], [], "o", markersize=ball_size, c=ball_color)
        
        # 3) TEXT ELEMENTS
        release_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes,
                                 color="red", fontsize=14, weight="bold")
        motion_text = ax.text2D(0.05, 0.90, "", transform=ax.transAxes,
                                color="blue", fontsize=12, weight="bold")
        distance_text = ax.text2D(0.05, 0.85, "", transform=ax.transAxes,
                                  color="green", fontsize=12, weight="bold")
        
        # 4) NEW: JOINT MARKERS
        # Create a set of all unique joint names from the connections list.
        unique_joints = set()
        for (joint_a, joint_b) in connections:
            unique_joints.add(joint_a)
            unique_joints.add(joint_b)
        joint_markers = {}
        for joint in unique_joints:
            marker_line, = ax.plot([], [], [], "o", color=player_color, markersize=5)
            joint_markers[joint] = marker_line

        if debug:
            logger.debug("Elements initialized (lines, ball, texts, and joint markers).")
        
        return lines, ball, release_text, motion_text, distance_text, joint_markers
    
    except Exception as e:
        logger.error(f"Error initializing elements: {e}")
        raise


def initialize_plot(zlim=20, elev=30, azim=60, figsize=(12, 10), debug=False):
    """
    Initialize a 3D plot with specified view settings and outputs setup details.

    Parameters:
    - zlim (float): The limit for the z-axis (height).
    - elev (float): Elevation angle in the z plane for the camera view.
    - azim (float): Azimuth angle in the x,y plane for the camera view.
    - figsize (tuple): Figure size.
    - debug (bool): Flag to enable debug logging.

    Returns:
    - fig: The Matplotlib figure object.
    - ax: The Matplotlib 3D axis object.
    """
    try:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_zlim([0, zlim])
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        if debug:
            logger.debug(f"Initialized 3D plot with Z limit: {zlim}, Elevation: {elev}, Azimuth: {azim}")
        return fig, ax
    except Exception as e:
        logger.error(f"Failed to initialize plot: {e}")
        raise

