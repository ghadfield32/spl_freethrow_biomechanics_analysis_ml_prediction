%%writefile animate_calc_bayes_shap.py
"""
Updated to unify calculated, bayesian, and shap feedback logic in update_meter_with_mode.
Introduces a polar subplot for the angle meter with angular ticks and a dynamic line graph.
"""

import logging
import pandas as pd 
import json
from IPython.display import HTML, display
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mplbasketball.court3d import Court3D, draw_court_3d
import matplotlib as mpl
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

# add for if we ever want to recalculate the shap min and max values
from ml.config.config_loader import load_config
from ml.config.config_models import AppConfig
from ml.shap.shap_utils import load_dataset, setup_logging, load_configuration, initialize_logger
from ml.shap.predict_with_shap_usage import predict_and_shap

from animate.court import draw_court, get_hoop_position
from animate.viewpoints import get_viewpoint 
from matplotlib.patches import Wedge

from animate.calc_bayes_shap_feature_engineering import automated_bayes_shap_summary
# Configure logging
logger = logging.getLogger('bayes_animate')
if not logger.hasHandlers():
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all logs
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Increase the animation embed limit to 30 MB
mpl.rcParams['animation.embed_limit'] = 30.0  # Value is in MB
# Global constant for converting degrees to radians
DEG_TO_RAD = np.pi / 180.0

# Define calculated metrics list
CALCULATED_METRICS = [
    'release_knee_angle',
    'release_wrist_angle',
    'release_elbow_angle',
    'wrist_max_angle',
    'elbow_max_angle',
    'knee_max_angle'
]


def display_separate_outputs(feedback_table, animation_html, feedback_mode):
    """
    Displays the feedback table and the animation in two separate outputs
    so they do not overlap in a single HTML container.
    
    Parameters:
      - feedback_table (pd.DataFrame): The feedback metrics table.
      - animation_html (IPython.display.HTML): The animation HTML object.
      - feedback_mode (str): The current feedback mode (for labeling purposes).
    
    This function is useful in Jupyter notebooks where the table might otherwise
    block interaction with the animation.
    """
    from IPython.display import display
    
    # Display the feedback table in its own block.
    print(f"Feedback Table ({feedback_mode.capitalize()} Mode):")
    display(feedback_table)
    
    # Display the animation in a separate output block.
    print("\nAnimation:")
    display(animation_html)



def load_bayesian_metrics_dict(
    json_path: str,
    debug: bool = False
) -> dict:
    """
    Load the Bayesian metrics dictionary from a JSON file.

    Parameters:
        json_path (str): Path to the JSON file.
        debug (bool): Whether to print debug information. Default is False.

    Returns:
        dict: Bayesian metrics dictionary without 'filter_name'.
    """
    try:
        with open(json_path, 'r') as f:
            bayesian_metrics_dict = json.load(f)
        
        if debug:
            logger.debug(f"Loaded Bayesian metrics dictionary from {json_path}")
            logger.debug(json.dumps(bayesian_metrics_dict, indent=4))
        return bayesian_metrics_dict
    except FileNotFoundError:
        logger.error(f"Bayesian metrics JSON file not found at {json_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"JSON decoding failed for file {json_path}")
        raise

def generate_feedback_table_all_metrics(feedback_mode, bayesian_metrics_dict, trial_data, debug=False):
    """
    Generates a comprehensive feedback table for all metrics based on the feedback mode.
    
    Parameters:
        feedback_mode (str): The mode of feedback ('bayesian', 'shap', 'calculated').
        bayesian_metrics_dict (dict): Dictionary containing Bayesian metrics for all selected metrics.
        trial_data (pd.DataFrame): DataFrame containing the trial data.
        debug (bool): If True, outputs debugging information.
    
    Returns:
        pd.DataFrame: A DataFrame containing the feedback data for all metrics, including the actual metric value
                      ('actual_value') for comparison with the classification.
    """
    try:
        feedback_records = []

        for metric_key, metric_info in bayesian_metrics_dict.items():
            # Use the metric key as the selected metric.
            selected_metric = metric_key  # e.g., 'release_ball_direction_x'
            filter_name = metric_info.get('filter_name', selected_metric)

            record = {
                'Metric_Key': metric_key,
                'Ongoing Metric Name': filter_name
            }
            if debug:
                logger.debug(f"Processing metric: {metric_key} (Filter: {filter_name})")
            
            # Define columns based on feedback mode.
            if feedback_mode.lower() == "bayesian":
                columns = {
                    'bayes_classification': f"{selected_metric}_bayes_classification",
                    'bayes_optimized': f"{selected_metric}_bayes_optimized",
                    'bayes_max': f"{selected_metric}_bayes_max",
                    'bayes_min': f"{selected_metric}_bayes_min",
                    'bayes_unit_change': f"{selected_metric}_bayes_unit_change",
                    'shap_importance': f"shap_{selected_metric}_importance"  # included even for bayesian
                }
            elif feedback_mode.lower() == "shap":
                columns = {
                    'shap_unit_change': f"shap_{selected_metric}_unit_change",
                    'shap_unit': f"shap_{selected_metric}_unit",
                    'shap_direction': f"shap_{selected_metric}_direction",
                    'shap_importance': f"shap_{selected_metric}_importance",
                    'shap_goal': f"shap_{selected_metric}_goal",
                    'shap_min': f"shap_{selected_metric}_min",
                    'shap_max': f"shap_{selected_metric}_max",
                    'shap_classification': f"shap_{selected_metric}_classification"
                }
            elif feedback_mode.lower() == "calculated":
                columns = {
                    'filtered_optimal_min': f"{selected_metric}_filtered_optimal_min",
                    'filtered_optimal_max': f"{selected_metric}_filtered_optimal_max",
                    'shot_classification': f"{selected_metric}_shot_classification"
                }
            else:
                raise ValueError("Invalid feedback mode selected.")

            # Extract values from trial_data for the defined columns.
            for key, col in columns.items():
                if col in trial_data.columns:
                    value = trial_data.at[0, col]
                    record[key] = value
                    if debug:
                        logger.debug(f"  {key}: {value}")
                else:
                    record[key] = np.nan
                    logger.warning(f"  Missing column: {col}")

            # NEW: Add the actual metric value from trial_data for comparison.
            if selected_metric in trial_data.columns:
                record['actual_value'] = trial_data.at[0, selected_metric]
                if debug:
                    logger.debug(f"  actual_value: {record['actual_value']}")
            else:
                record['actual_value'] = np.nan
                logger.warning(f"  Missing ongoing metric column: {selected_metric}")
            
            feedback_records.append(record)

        # Create DataFrame from records
        feedback_df = pd.DataFrame(feedback_records)

        # NEW FEATURE: Ensure 'shap_importance' column exists.
        if 'shap_importance' not in feedback_df.columns:
            # If the column is missing, add it with default NaN values.
            feedback_df['shap_importance'] = np.nan
            if debug:
                logger.debug("Added missing 'shap_importance' column with default NaN values.")
        else:
            # Convert to numeric if possible so that sorting works correctly.
            feedback_df['shap_importance'] = pd.to_numeric(feedback_df['shap_importance'], errors='coerce')

        # NEW FEATURE: Sort the feedback table by 'shap_importance'
        feedback_df = feedback_df.sort_values(by='shap_importance', na_position='last', ascending=False)
        if debug:
            logger.debug("Feedback table sorted by 'shap_importance'.")
            logger.debug(feedback_df.head())

        return feedback_df

    except Exception as e:
        logger.error(f"Error in generate_feedback_table_all_metrics: {e}")
        raise





def add_bayes_optimal_lines(
    ax: plt.Axes,
    min_angle: float,
    max_angle: float,
    feedback_mode: str = None,  # new
    debug: bool = False
) -> None:
    """
    Add optimal min and max lines to the angle meter, including release min and max angles.
    """
    try:
        # Convert angles to radians
        min_angle_rad = min_angle * np.pi / 180
        max_angle_rad = max_angle * np.pi / 180

        # Check if lines already exist
        if not hasattr(ax, 'optimal_min_line'):
            ax.optimal_min_line, = ax.plot(
                [0, min_angle_rad], [0, 1], color='blue', lw=2, linestyle="--", label='Bayes Min'
            )
        if not hasattr(ax, 'optimal_max_line'):
            ax.optimal_max_line, = ax.plot(
                [0, max_angle_rad], [0, 1], color='green', lw=2, linestyle="--", label='Bayes Max'
            )

        if debug:
            logger.debug(
                f"Added optimal lines at {min_angle}°, {max_angle}° with labels 'Bayes Min' and 'Bayes Max'."
            )
    except Exception as e:
        logger.error(f"Error in add_bayes_optimal_lines: {e}")
        raise

def add_bayes_optimal_lines_to_bar(ax_bar, min_val, max_val, selected_metric: str, feedback_mode: str, debug=False):
    try:
        # Determine the labels based on the feedback mode:
        if feedback_mode.lower() == "shap":
            min_label = f"{selected_metric} SHAP Min"
            max_label = f"{selected_metric} SHAP Max"
        elif feedback_mode.lower() == "calculated":
            min_label = f"{selected_metric} Calc Min"
            max_label = f"{selected_metric} Calc Max"
        else:  # Assume bayesian or default
            min_label = f"{selected_metric} Bayes Min"
            max_label = f"{selected_metric} Bayes Max"

        if not hasattr(ax_bar, 'bar_min_line'):
            ax_bar.bar_min_line = ax_bar.axvline(x=min_val, color='blue', lw=2, linestyle='--', label=min_label)
        if not hasattr(ax_bar, 'bar_max_line'):
            ax_bar.bar_max_line = ax_bar.axvline(x=max_val, color='green', lw=2, linestyle='--', label=max_label)
        if debug:
            logger.debug(f"Added bar optimal lines at {min_val} and {max_val} with labels '{min_label}' and '{max_label}'.")
    except Exception as e:
        logger.error(f"Error in add_bayes_optimal_lines_to_bar: {e}")
        raise


def initialize_bayes_elements(
    ax: plt.Axes,
    connections: list,
    player_color: str,
    player_lw: float,
    ball_color: str,
    ball_size: float,
    debug: bool = False
) -> (dict, plt.Line2D, plt.Text, plt.Text, plt.Text):
    """
    Initialize plot elements such as lines for the player skeleton and the ball.
    """
    try:
        # Initialize lines for each body connection
        lines = {connection: ax.plot([], [], [], c=player_color, lw=player_lw)[0] for connection in connections}
        
        # Initialize the ball as a point
        ball, = ax.plot([], [], [], "o", markersize=ball_size, c=ball_color)
        
        # Text elements for annotations
        release_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, color="red", fontsize=14, weight="bold")
        motion_text = ax.text2D(0.05, 0.90, "", transform=ax.transAxes, color="blue", fontsize=12, weight="bold")
        
        # New text element for distance (positioned below motion_text)
        distance_text = ax.text2D(0.05, 0.85, "", transform=ax.transAxes, color="green", fontsize=12, weight="bold")
        
        if debug:
            logger.debug("Elements initialized (lines, ball, text labels).")
        
        return lines, ball, release_text, motion_text, distance_text
    except Exception as e:
        logger.error(f"Error initializing elements: {e}")
        raise

def update_bayes_frame(
    ax: plt.Axes,
    frame: int,
    df: pd.DataFrame,
    release_frame: int,
    lines: dict,
    ball: plt.Line2D,
    release_text: plt.Text,
    motion_text: plt.Text,
    connections: list,
    ball_color: str,
    highlight_color: str,
    debug: bool = False,
    selected_metric: str = None,
    selected_metric_filter_name: str = None,
    joint_markers: dict = None
):
    """
    Update the player's skeleton, ball position, text annotations, and joint markers.
    If joint_markers is provided, each joint marker is updated with the current frame's x, y, z data.
    """
    try:
        # Update skeleton lines
        for connection in connections:
            joint_start, joint_end = connection
            x_start = df.at[frame, f"{joint_start}_x"]
            y_start = df.at[frame, f"{joint_start}_y"]
            z_start = df.at[frame, f"{joint_start}_z"]
            x_end = df.at[frame, f"{joint_end}_x"]
            y_end = df.at[frame, f"{joint_end}_y"]
            z_end = df.at[frame, f"{joint_end}_z"]
            lines[connection].set_data([x_start, x_end], [y_start, y_end])
            lines[connection].set_3d_properties([z_start, z_end])
        
        # Update ball position
        ball_x = df.at[frame, "ball_x"]
        ball_y = df.at[frame, "ball_y"]
        ball_z = df.at[frame, "ball_z"]
        ball.set_data([ball_x], [ball_y])
        ball.set_3d_properties([ball_z])
        
        # Update text annotations
        if frame == release_frame:
            release_text.set_text("Release Point")
            release_text.set_color(highlight_color)
        else:
            release_text.set_text("")
        
        if df.at[frame, "shooting_motion"] == 1:
            motion_text.set_text("Shooting Motion")
        else:
            motion_text.set_text("")
        
        # NEW: Update joint markers if provided
        if joint_markers is not None:
            for joint, marker_line in joint_markers.items():
                x_joint = df.at[frame, f"{joint}_x"]
                y_joint = df.at[frame, f"{joint}_y"]
                z_joint = df.at[frame, f"{joint}_z"]
                marker_line.set_data([x_joint], [y_joint])
                marker_line.set_3d_properties([z_joint])
        
        if debug:
            logger.debug(f"Updated frame {frame}: Player, ball, texts, and joint markers updated.")
    
    except Exception as e:
        logger.error(f"Error updating frame {frame}: {e}")
        raise




def update_meter_with_mode(
    ax_meter,
    df,
    frame,
    needle,
    one_liner,  # single text element for combined information
    angle_key: str,
    feedback_key: str,
    feedback_mode: str = "calculated",
    bayesian_metrics_dict: dict = None,
    debug: bool = False,
    conditional_max: bool = False
):
    """
    Update the polar angle meter based on the chosen feedback mode.
    This version updates the dynamic text element (one_liner) and rotates the needle.
    The gauge_fill wedge (red fill) is updated to span from 0° to the current angle.
    
    Efficiency improvements:
      - Uses the global DEG_TO_RAD constant.
      - Minimizes repeated conversions.
      - Re-calls legend only on frame 0.
    
    Parameters:
      - ax_meter: The polar axes.
      - df: DataFrame with the current frame data.
      - frame: The current frame index.
      - needle: The dynamic needle line.
      - one_liner: The text element for displaying combined information.
      - angle_key: Column name for the ongoing angle.
      - feedback_key: Key used to look up the metric.
      - feedback_mode: "calculated", "bayesian", or "shap".
      - bayesian_metrics_dict: Dictionary with metric info.
      - debug: If True, print debug information.
      - conditional_max: If True, apply conditional logic for max lines.
    """
    try:
        # Get the current angle (in degrees) and convert to radians using the precomputed constant.
        current_angle_degs = df.loc[frame, angle_key]
        rad = current_angle_degs * DEG_TO_RAD
        # Update the needle to point at the current angle.
        needle.set_data([rad, rad], [0, 0.8])
        
        # Update the gauge fill wedge so that its theta2 equals the current angle.
        if hasattr(ax_meter, 'gauge_fill'):
            ax_meter.gauge_fill.theta2 = current_angle_degs
            if debug:
                print(f"Gauge fill updated: theta2 set to {current_angle_degs}°.")
        
        # Build the info string for the one-liner text.
        info = f"Ongoing {angle_key}: {current_angle_degs:.1f}°"
        
        # --- Feedback mode-specific logic (calculated, bayesian, shap) remains unchanged ---
        if feedback_mode.lower() == "calculated":
            if selected_metric := feedback_key:  # fallback to selected metric name
                if selected_metric in df.columns:
                    calc_min_col = f"{feedback_key}_filtered_optimal_min"
                    calc_max_col = f"{feedback_key}_filtered_optimal_max"
                    calc_min = df.loc[frame, calc_min_col] if calc_min_col in df.columns else np.nan
                    calc_max = df.loc[frame, calc_max_col] if calc_max_col in df.columns else np.nan
                    if debug:
                        print(f"Using fallback columns: {calc_min_col} and {calc_max_col}.")
            else:
                raise ValueError("Selected metric not found in DataFrame.")
            
            if not hasattr(ax_meter, 'calc_min_line'):
                calc_min_rad = calc_min * DEG_TO_RAD
                ax_meter.calc_min_line, = ax_meter.plot([0, calc_min_rad], [0, 1],
                                                        color='blue', lw=2, linestyle='--',
                                                        label=f"{angle_key} Calc Min")
                if debug:
                    print(f"Created Calculated Min line at {calc_min:.1f}°.")
            else:
                calc_min_rad = calc_min * DEG_TO_RAD
                ax_meter.calc_min_line.set_data([0, calc_min_rad], [0, 1])
                if debug:
                    print(f"Updated Calculated Min line to {calc_min:.1f}°.")
            if not hasattr(ax_meter, 'calc_max_line'):
                calc_max_rad = calc_max * DEG_TO_RAD
                if (not conditional_max) or (current_angle_degs >= calc_max):
                    ax_meter.calc_max_line, = ax_meter.plot([0, calc_max_rad], [0, 1],
                                                            color='green', lw=2, linestyle='--',
                                                            label=f"{angle_key} Calc Max")
                    if debug:
                        print(f"Created Calculated Max line at {calc_max:.1f}°.")
            else:
                calc_max_rad = calc_max * DEG_TO_RAD
                if (not conditional_max) or (current_angle_degs >= calc_max):
                    ax_meter.calc_max_line.set_data([0, calc_max_rad], [0, 1])
                    if debug:
                        print(f"Updated Calculated Max line to {calc_max:.1f}°.")
            
            shot_class_col = f"{feedback_key}_shot_classification"
            if shot_class_col in df.columns:
                shot_classification = df.loc[frame, shot_class_col]
                info += f" | Shot Class: {shot_classification}"
            else:
                info += " | Shot Class: N/A"
        
        elif feedback_mode.lower() == "bayesian":
            bayes_min_col = f"{feedback_key}_bayes_min"
            bayes_max_col = f"{feedback_key}_bayes_max"
            bayes_class_col = f"{feedback_key}_bayes_classification"
            if bayes_min_col in df.columns:
                bayes_min = df.loc[frame, bayes_min_col]
            else:
                bayes_min = bayesian_metrics_dict.get(feedback_key.lower(), {}).get("bayes_min", 0)
            if bayes_max_col in df.columns:
                bayes_max = df.loc[frame, bayes_max_col]
            else:
                bayes_max = bayesian_metrics_dict.get(feedback_key.lower(), {}).get("bayes_max", 180)
            if not hasattr(ax_meter, 'bayes_min_line'):
                ax_meter.bayes_min_line, = ax_meter.plot(
                    [0, bayes_min * DEG_TO_RAD], [0, 1],
                    color='blue', lw=2, linestyle='--', label=f"{angle_key} Bayes Min"
                )
                if debug:
                    print(f"Created Bayes Min line at {bayes_min:.1f}°.")
            else:
                ax_meter.bayes_min_line.set_data([0, bayes_min * DEG_TO_RAD], [0, 1])
                if debug:
                    print(f"Updated Bayes Min line to {bayes_min:.1f}°.")
            if (not conditional_max) or (current_angle_degs >= bayes_max):
                if not hasattr(ax_meter, 'bayes_max_line'):
                    ax_meter.bayes_max_line, = ax_meter.plot(
                        [0, bayes_max * DEG_TO_RAD], [0, 1],
                        color='green', lw=2, linestyle='--', label=f"{angle_key} Bayes Max"
                    )
                    if debug:
                        print(f"Created Bayes Max line at {bayes_max:.1f}°.")
                else:
                    ax_meter.bayes_max_line.set_data([0, bayes_max * DEG_TO_RAD], [0, 1])
                    if debug:
                        print(f"Updated Bayes Max line to {bayes_max:.1f}°.")
            if bayes_class_col in df.columns:
                bayes_classification = df.loc[frame, bayes_class_col]
            else:
                bayes_classification = bayesian_metrics_dict.get(feedback_key.lower(), {}).get("bayes_classification", "N/A")
            info += f" | Bayes Class: {bayes_classification}"
        
        elif feedback_mode.lower() == "shap":
            selected_metric = feedback_key  # for consistency
            shap_class_col = f"shap_{selected_metric}_classification"
            shap_unit_change_col = f"shap_{selected_metric}_unit_change"
            shap_unit_col = f"shap_{selected_metric}_unit"
            shap_direction_col = f"shap_{selected_metric}_direction"
            shap_imp_col = f"shap_{selected_metric}_importance"
            shap_class = df.loc[frame, shap_class_col] if shap_class_col in df.columns else "N/A"
            shap_unit_change = df.loc[frame, shap_unit_change_col] if shap_unit_change_col in df.columns else "N/A"
            shap_unit = df.loc[frame, shap_unit_col] if shap_unit_col in df.columns else ""
            shap_direction = df.loc[frame, shap_direction_col] if shap_direction_col in df.columns else "N/A"
            if shap_imp_col in df.columns:
                shap_imp = df.loc[frame, shap_imp_col]
                try:
                    shap_importance = f"{float(shap_imp):.3f}"
                except (ValueError, TypeError):
                    shap_importance = shap_imp
            else:
                shap_importance = "N/A"
            shap_min_col = f"shap_{selected_metric}_min"
            shap_max_col = f"shap_{selected_metric}_max"
            if shap_min_col in df.columns:
                shap_min = df.loc[frame, shap_min_col]
            else:
                shap_min = 0
            if shap_max_col in df.columns:
                shap_max = df.loc[frame, shap_max_col]
            else:
                shap_max = 180
            shap_min_rad = shap_min * DEG_TO_RAD
            shap_max_rad = shap_max * DEG_TO_RAD
            if not hasattr(ax_meter, 'shap_min_line'):
                ax_meter.shap_min_line, = ax_meter.plot(
                    [0, shap_min_rad], [0, 1],
                    color='blue', lw=2, linestyle='--', label=f"{angle_key} SHAP Min"
                )
            else:
                ax_meter.shap_min_line.set_data([0, shap_min_rad], [0, 1])
            if (not conditional_max) or (current_angle_degs >= shap_max):
                if not hasattr(ax_meter, 'shap_max_line'):
                    ax_meter.shap_max_line, = ax_meter.plot(
                        [0, shap_max_rad], [0, 1],
                        color='green', lw=2, linestyle='--', label=f"{angle_key} SHAP Max"
                    )
                else:
                    ax_meter.shap_max_line.set_data([0, shap_max_rad], [0, 1])
            one_liner_str = (f"Ongoing {angle_key}: {df.loc[frame, angle_key]:.1f}° | "
                             f"SHAP Class: {shap_class} |\n "
                             f"Direction: {shap_direction} | SHAP Imp: {shap_importance}")
        
        else:
            logger.warning(f"Unknown feedback mode: {feedback_mode}")
            info += " | Class: N/A | SHAP: N/A | Direction: N/A"
            if debug:
                print("Set default one-liner info to N/A for unknown feedback mode.")
        
        # Set the one-liner text based on the mode.
        one_liner.set_text(info if feedback_mode.lower() != "shap" else one_liner_str)
        
        if debug:
            print(f"Frame={frame}, feedback_mode='{feedback_mode}' => {one_liner.get_text()}")
        
        # Re-call legend on frame 0 to include any new lines.
        if frame == 0:
            ax_meter.legend(loc='upper right')
    
    except Exception as e:
        logger.error(f"Error in update_meter_with_mode: {e}")
        raise













def initialize_bar_meter(ax_bar, min_val, max_val):
    """
    Initializes a horizontal bar to represent the angle from min_val to max_val.
    Returns the bar container so you can update it each frame.
    """
    ax_bar.set_xlim([min_val, max_val])
    ax_bar.set_ylim([-0.5, 0.5])  # Just enough space for a single bar
    ax_bar.set_xlabel("Angle (degrees)")
    ax_bar.set_yticks([])

    # Draw a single bar initially at zero
    bar_container = ax_bar.barh(
        y=0, width=0, height=0.4, color='red', align='center'
    )

    return bar_container

def update_bar_meter(bar_container, current_angle, min_val=0, max_val=180):
    """
    Updates the bar's width so that it visually represents the current_angle.
    """
    # bar_container is usually a list with a single bar (bar_container[0])
    bar = bar_container[0]
    
    # Constrain/clip angle if needed (optional).
    clipped_angle = max(min_val, min(current_angle, max_val))
    
    # Update the bar width
    bar.set_width(clipped_angle)



def initialize_line_graph(
    ax_line,
    static_min: float,
    static_max: float,
    selected_metric_filter_name: str,
    selected_metric: str,
    feedback_mode: str,
    debug=False
):
    """
    Initialize the line graph subplot with dynamic metric tracking.
    The dynamic (ongoing) metric line is drawn in red.
    
    CHANGES:
      - The title now includes the metric name.
      - The red ongoing line’s label is now set to the metric name in lowercase.
      - Static min and max lines are labeled based on the chosen feedback mode.
    """
    try:
        ax_line.set_title(f"Metric Over Time: {selected_metric}", fontsize=14, pad=20)
        ax_line.set_xlabel("Frame", fontsize=12)
        ax_line.set_ylabel(f"{selected_metric} (degrees)", fontsize=12)

        # Dynamic (ongoing) metric line in red with a lowercase label
        line_metric, = ax_line.plot([], [], lw=2, color='red',
                                    label=f"{selected_metric.lower()} Ongoing")  # <--- CHANGED here!

        # Determine labels for the static min/max lines based on feedback mode
        if feedback_mode.lower() == "shap":
            min_label = f"{selected_metric} SHAP Min"
            max_label = f"{selected_metric} SHAP Max"
        elif feedback_mode.lower() == "calculated":
            min_label = f"{selected_metric} Calc Min"
            max_label = f"{selected_metric} Calc Max"
        else:  # bayesian or default
            min_label = f"{selected_metric} Bayes Min"
            max_label = f"{selected_metric} Bayes Max"

        # Draw static min and max lines
        line_min = ax_line.axhline(y=static_min, color='blue', linestyle='--', label=min_label)
        line_max = ax_line.axhline(y=static_max, color='green', linestyle='--', label=max_label)

        # Optional trial max line (initially invisible)
        line_trial_max, = ax_line.plot([], [], lw=2, color='red', linestyle='--',
                                       label=f"{selected_metric} Trial Max")
        line_trial_max.set_visible(False)

        data_frames = []
        data_values = []

        # Set initial y-limits with a small padding
        initial_lower = static_min - 5
        initial_upper = static_max + 5
        ax_line.set_ylim(initial_lower, initial_upper)

        # Create the legend
        ax_line.legend(loc='upper right')

        if debug:
            logger.debug("Line graph initialized with dynamic metric tracking.")

        return {
            'line_metric': line_metric,
            'line_min': line_min,
            'line_max': line_max,
            'line_trial_max': line_trial_max,
            'data_frames': data_frames,
            'data_values': data_values,
            'current_ymin': initial_lower,
            'current_ymax': initial_upper,
            'current_trial_max': static_max,
            'trial_max_reached': False
        }
    except Exception as e:
        logger.error(f"Error initializing line graph: {e}")
        raise


def animate_trial_with_calc_bayes_shap_angle_meter(
    df: pd.DataFrame,
    release_frame: int,
    selected_metric: str,
    bayesian_metrics_dict: dict,
    feedback_mode: str = "bayesian",
    viewpoint_name: str = "side_view_right",
    connections: list = None,
    zlim: float = 15.0,
    player_color: str = "purple",
    player_lw: float = 2.0,
    ball_color: str = "#ee6730",
    ball_size: float = 20.0,
    highlight_color: str = "red",
    show_court: bool = True,
    court_type: str = "nba",
    units: str = "ft",
    debug: bool = False,
    frames_to_animate: list = None,
    show_selected_metric: bool = False,  
    polar_plot: bool = True,
    bar_plot: bool = True,
    line_plot: bool = True,
    save_path: Optional[str] = None,
    notebook_mode: bool = False,
    figure_width: float = 6,
    height_3d: float = 7.0,
    height_polar: float = 5.0,
    height_bar: float = 2.0,
    height_line: float = 2.0,
    space_3d_polar: float = 0.00001,    # vertical gap between 3D and polar
    space_polar_bar: float = 0.00001,    # vertical gap between polar and bar
    space_bar_line: float = 1.0          # vertical gap between bar and line
) -> HTML:
    """
    Animate a basketball trial with integrated angle/line/polar/bar subplots.
    
    Additional efficiency improvements:
      - Caches frequently accessed DataFrame columns as NumPy arrays.
      - Precomputes frequently built column names.
      - Updates dynamic axis limits for the line graph only every 10 frames.
      - Uses the global DEG_TO_RAD constant.
    
    New: Independent vertical spacing between subplots.
    """
    try:
        from IPython.display import HTML
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.gridspec import GridSpec
        from matplotlib.lines import Line2D
        from matplotlib.patches import Wedge

        if connections is None:
            raise ValueError("connections list cannot be None.")

        if debug:
            logger.debug("Starting animation setup.")
            logger.debug(f"Total frames in df: {len(df)}; Release frame: {release_frame}")
            logger.debug(f"Viewpoint: {viewpoint_name}, Metric: {selected_metric}")

        # Get viewpoint information.
        viewpoint = get_viewpoint(viewpoint_name)
        elev, azim = viewpoint["elev"], viewpoint["azim"]

        # Build height ratios for the figure layout.
        height_ratios = []
        height_ratios.append(height_3d)
        if polar_plot:
            height_ratios.append(space_3d_polar)
            height_ratios.append(height_polar)
        if bar_plot:
            gap_to_use = space_polar_bar if polar_plot else space_3d_polar
            height_ratios.append(gap_to_use)
            height_ratios.append(height_bar)
        if line_plot:
            gap_to_use = space_bar_line if bar_plot else (space_polar_bar if polar_plot else space_3d_polar)
            height_ratios.append(gap_to_use)
            height_ratios.append(height_line)
        figure_height = sum(height_ratios)

        # Create the figure.
        fig = plt.figure(figsize=(figure_width, figure_height))
        fig.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0, rect=[0, 0, 1, 1])
        row_count = len(height_ratios)
        gs = GridSpec(row_count, 1, figure=fig, height_ratios=height_ratios)

        # Place the subplots.
        row_idx = 0
        ax_3d = fig.add_subplot(gs[row_idx, 0], projection='3d')
        ax_3d.view_init(elev=elev, azim=azim)
        ax_3d.set_zlim([0, zlim])
        ax_3d.set_box_aspect((2, 2, 1))

        ax_meter = None
        if polar_plot:
            row_idx += 2  # Skip gap row and then add polar subplot.
            ax_meter = fig.add_subplot(gs[row_idx, 0], polar=True)

        ax_bar = None
        if bar_plot:
            row_idx += 2  # Skip gap and add bar subplot.
            ax_bar = fig.add_subplot(gs[row_idx, 0])

        ax_line = None
        if line_plot:
            row_idx += 2  # Skip gap and add line subplot.
            ax_line = fig.add_subplot(gs[row_idx, 0])

        # Draw the court if enabled.
        if show_court:
            draw_court(ax_3d, court_type=court_type, units=units, debug=debug)
            _hoopx, _hoopy, _hoopz = get_hoop_position(court_type=court_type, units=units, debug=debug)

        # Initialize 3D elements.
        lines, ball, release_text, motion_text, distance_text = initialize_bayes_elements(
            ax=ax_3d,
            connections=connections,
            player_color=player_color,
            player_lw=player_lw,
            ball_color=ball_color,
            ball_size=ball_size,
            debug=debug
        )

        # Retrieve selected metric filter name.
        metric_info = bayesian_metrics_dict.get(selected_metric.lower())
        if not metric_info:
            raise ValueError(f"Metric '{selected_metric}' not found in bayesian_metrics_dict.")
        selected_metric_filter_name = metric_info.get("filter_name", selected_metric)

        # Determine static min/max based on feedback mode.
        if feedback_mode.lower() == "shap":
            shap_min_col = f"shap_{selected_metric}_min"
            shap_max_col = f"shap_{selected_metric}_max"
            static_min = df.at[0, shap_min_col] if shap_min_col in df.columns else 0
            static_max = df.at[0, shap_max_col] if shap_max_col in df.columns else 180
        elif feedback_mode.lower() == "calculated":
            calc_min_col = f"{selected_metric}_filtered_optimal_min"
            calc_max_col = f"{selected_metric}_filtered_optimal_max"
            static_min = df.at[0, calc_min_col] if calc_min_col in df.columns else 0
            static_max = df.at[0, calc_max_col] if calc_max_col in df.columns else 180
        else:
            static_min = metric_info.get("bayes_min", 0)
            static_max = metric_info.get("bayes_max", 180)

        # ---------- POLAR METER SETUP ----------
        angle_meter_obj = None
        if polar_plot and ax_meter is not None:
            for deg in range(0, 181, 30):
                deg_rad = np.deg2rad(deg)
                ax_meter.plot([deg_rad, deg_rad], [0, 1],
                              color='gray', linewidth=0.5, alpha=0.4, zorder=0)
            if not hasattr(ax_meter, 'gauge_fill'):
                ax_meter.gauge_fill = Wedge(center=(0, 0), r=1.0, theta1=0, theta2=0,
                                            facecolor='red', alpha=0.3, transform=ax_meter.transData)
                ax_meter.add_patch(ax_meter.gauge_fill)
            needle, = ax_meter.plot([0, 0], [0, 0.8],
                                    lw=3, color='red',
                                    label=f"{selected_metric.lower()} Ongoing")
            ax_meter.gauge_fill.set_zorder(1)
            needle.set_zorder(2)
            ax_meter.set_title(f"Polar Angle Meter: {selected_metric}", fontsize=14, pad=20)
            one_liner = ax_meter.text(0.5, 0.25, "", transform=ax_meter.transAxes,
                                      ha='center', va='center', fontsize=10, color='red', zorder=12)
            ax_meter.set_theta_offset(np.pi)
            ax_meter.set_theta_direction(-1)
            angle_ticks = np.linspace(0, np.pi, 6)
            angle_labels = [f'{int(deg)}°' for deg in np.linspace(0, 180, 6)]
            ax_meter.set_xticks(angle_ticks)
            ax_meter.set_xticklabels(angle_labels)
            ax_meter.set_yticklabels([])
            ax_meter.grid(False)
            ax_meter.set_ylim(0, 1)
            ax_meter.fill_between(np.linspace(np.pi, 2*np.pi, 100), 0, 1, color="white", zorder=10)
            ax_meter.spines['polar'].set_visible(False)
            ax_meter.plot([0, np.pi], [0, 1], color='black', lw=1)
            if show_selected_metric:
                selected_value = df[selected_metric].iloc[0]
                selected_value_rad = selected_value * DEG_TO_RAD
                if not hasattr(ax_meter, 'selected_metric_line'):
                    ax_meter.selected_metric_line, = ax_meter.plot(
                        [0, selected_value_rad], [0, 1],
                        color='black', lw=2, linestyle='-', label=f"{selected_metric} Selected"
                    )
            ax_meter.legend(loc='upper right')
            angle_meter_obj = {'ax_meter': ax_meter, 'needle': needle, 'one_liner': one_liner}

        # ---------- BAR METER SETUP ----------
        bar_container = None
        bar_ongoing_text = None
        if bar_plot and ax_bar is not None:
            ax_bar.set_title(f"Bar Meter: {selected_metric}", fontsize=14, pad=30)
            bar_ongoing_text = ax_bar.text(0.5, 1.02, "", transform=ax_bar.transAxes,
                                           ha='center', va='bottom', fontsize=10, color='red')
            bar_container = initialize_bar_meter(ax_bar, min_val=0, max_val=180)
            add_bayes_optimal_lines_to_bar(ax_bar, static_min, static_max, 
                                           selected_metric_filter_name, feedback_mode, debug)
            bar_container[0].set_label(f"{selected_metric.lower()} Ongoing")
            first_val = df[selected_metric].iloc[0]
            if not hasattr(ax_bar, 'selected_metric_line'):
                ax_bar.selected_metric_line = ax_bar.axvline(
                    x=first_val, color='black', lw=2, linestyle='-',
                    label=f"{selected_metric} Selected"
                )
            bar_handles = [bar_container[0]]
            if hasattr(ax_bar, 'bar_min_line'):
                bar_handles.append(ax_bar.bar_min_line)
            if hasattr(ax_bar, 'bar_max_line'):
                bar_handles.append(ax_bar.bar_max_line)
            if hasattr(ax_bar, 'selected_metric_line'):
                bar_handles.append(ax_bar.selected_metric_line)
            ax_bar.legend(handles=bar_handles, loc='upper right')

        # ---------- LINE GRAPH SETUP ----------
        line_graph_obj = None
        line_ongoing_text = None
        if line_plot and ax_line is not None:
            line_graph_obj = initialize_line_graph(ax_line, static_min, static_max,
                                                   selected_metric_filter_name, selected_metric,
                                                   feedback_mode, debug)
            line_ongoing_text = ax_line.text(0.5, 1.02, "", transform=ax_line.transAxes,
                                             ha='center', va='bottom', fontsize=10, color='red')
            if show_selected_metric:
                selected_value = df[selected_metric].iloc[0]
                line_graph_obj['selected_metric_line'] = ax_line.axhline(
                    y=selected_value, color='black', linestyle='-', lw=2, label=f"{selected_metric_filter_name} Selected"
                )
                line_handles = [
                    line_graph_obj['line_min'],
                    line_graph_obj['line_max'],
                    line_graph_obj['line_metric'],
                    line_graph_obj['selected_metric_line']
                ]
                ax_line.legend(handles=line_handles, loc='upper right')

        # ---------- 3D Legend ----------
        player_handle = Line2D([0], [0], color=player_color, lw=player_lw, label='Player')
        ball_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=ball_color, label='Ball')
        ax_3d.legend(handles=[player_handle, ball_handle], loc='upper right')

        if release_frame < 0 or release_frame >= len(df):
            release_frame = 0
        if debug:
            logger.debug(f"Final release_frame used: {release_frame}")

        if 'release_point_filter' in df.columns and release_frame < len(df):
            trial_release = (
                df.loc[release_frame, selected_metric_filter_name] 
                if df.loc[release_frame, 'release_point_filter'] == 1 else None
            )
        else:
            trial_release = None

        # ---------- Cache Frequently Accessed Data ----------
        # Cache the selected metric column as a NumPy array.
        selected_metric_values = df[selected_metric_filter_name].to_numpy()
        # Optionally, cache additional columns if used frequently.
        distance_to_basket = df['distance_to_basket'].to_numpy() if 'distance_to_basket' in df.columns else None
        shooting_motion = df['shooting_motion'].to_numpy() if 'shooting_motion' in df.columns else None

        # Precompute frequently built column names and store in a dictionary.
        col_names = {}
        if feedback_mode.lower() == "bayesian":
            col_names["bayes_min"] = f"{selected_metric}_bayes_min"
            col_names["bayes_max"] = f"{selected_metric}_bayes_max"
            col_names["bayes_class"] = f"{selected_metric}_bayes_classification"
            col_names["bayes_unit"] = f"{selected_metric}_bayes_unit_change"
        elif feedback_mode.lower() == "shap":
            col_names["shap_class"] = f"shap_{selected_metric}_classification"
            col_names["shap_unit"] = f"shap_{selected_metric}_unit_change"
            col_names["shap_direction"] = f"shap_{selected_metric}_direction"
            col_names["shap_imp"] = f"shap_{selected_metric}_importance"
        elif feedback_mode.lower() == "calculated":
            col_names["shot_class"] = f"{selected_metric}_shot_classification"

        # ---------- Update Function ----------
        def update_func(frame: int):
            # Update 3D elements.
            update_bayes_frame(
                ax=ax_3d,
                frame=frame,
                df=df,
                release_frame=release_frame,
                lines=lines,
                ball=ball,
                release_text=release_text,
                motion_text=motion_text,
                connections=connections,
                ball_color=ball_color,
                highlight_color=highlight_color,
                debug=debug,
                selected_metric=selected_metric,
                selected_metric_filter_name=selected_metric_filter_name
            )
            # Update polar meter.
            if polar_plot and angle_meter_obj is not None:
                update_meter_with_mode(
                    ax_meter=angle_meter_obj['ax_meter'],
                    df=df,
                    frame=frame,
                    needle=angle_meter_obj['needle'],
                    one_liner=angle_meter_obj['one_liner'],
                    angle_key=selected_metric_filter_name,
                    feedback_key=selected_metric,
                    feedback_mode=feedback_mode,
                    bayesian_metrics_dict=bayesian_metrics_dict,
                    debug=debug
                )
                # if show_selected_metric:
                #     current_val = selected_metric_values[frame]
                #     current_rad = current_val * DEG_TO_RAD
                #     if hasattr(angle_meter_obj['ax_meter'], 'selected_metric_line'):
                #         angle_meter_obj['ax_meter'].selected_metric_line.set_data([current_rad, current_rad], [0, 1])
            # Update bar meter.
            current_val = selected_metric_values[frame]
            if bar_container is not None:
                update_bar_meter(bar_container, current_angle=current_val, min_val=0, max_val=180)

            # Update one-liner text based on feedback mode.
            if feedback_mode.lower() == "bayesian":
                bayes_class = df.loc[frame, col_names.get("bayes_class", "N/A")] if col_names.get("bayes_class") in df.columns else "N/A"
                bayes_unit = df.loc[frame, col_names.get("bayes_unit", "N/A")] if col_names.get("bayes_unit") in df.columns else "N/A"
                one_liner_str = (f"Ongoing {selected_metric_filter_name}: {current_val:.1f}° | "
                                 f"Bayes Class: {bayes_class} | Unit Change: {bayes_unit}")
            elif feedback_mode.lower() == "shap":
                shap_class = df.loc[frame, col_names.get("shap_class", "N/A")] if col_names.get("shap_class") in df.columns else "N/A"
                shap_unit = df.loc[frame, col_names.get("shap_unit", "N/A")] if col_names.get("shap_unit") in df.columns else ""
                shap_direction = df.loc[frame, col_names.get("shap_direction", "N/A")] if col_names.get("shap_direction") in df.columns else "N/A"
                if col_names.get("shap_imp") in df.columns:
                    shap_imp = df.loc[frame, col_names.get("shap_imp")]
                    try:
                        shap_importance = f"{float(shap_imp):.3f}"
                    except:
                        shap_importance = shap_imp
                else:
                    shap_importance = "N/A"
                one_liner_str = (
                    f"Ongoing {selected_metric_filter_name}: {current_val:.1f}° | "
                    f"SHAP Class: {shap_class} |\n Direction: {shap_direction} | SHAP Imp: {shap_importance}"
                )
            elif feedback_mode.lower() == "calculated":
                shot_class = df.loc[frame, col_names.get("shot_class", "N/A")] if col_names.get("shot_class") in df.columns else "N/A"
                one_liner_str = (f"Ongoing {selected_metric_filter_name}: {current_val:.1f}° | "
                                 f"Shot Class: {shot_class}")
            else:
                one_liner_str = f"Ongoing {selected_metric_filter_name}: {current_val:.1f}°"

            if bar_ongoing_text:
                bar_ongoing_text.set_text(one_liner_str)

            if distance_to_basket is not None:
                dist = distance_to_basket[frame]
                distance_text.set_text(f"Distance to Basket: {dist:.2f} ft" if not np.isnan(dist) else "")
            else:
                distance_text.set_text("")

            # Update the line graph; update y–axis limits only every 10 frames.
            if line_graph_obj is not None:
                frame_num = frame
                line_graph_obj['data_frames'].append(frame_num)
                line_graph_obj['data_values'].append(current_val)
                line_graph_obj['line_metric'].set_data(line_graph_obj['data_frames'],
                                                       line_graph_obj['data_values'])
                ax_line.set_xlim(left=0, right=max(line_graph_obj['data_frames']) + 1)
                if frame % 10 == 0:
                    cmin = min(line_graph_obj['data_values']) - 5
                    cmax = max(line_graph_obj['data_values']) + 5
                    if cmin < line_graph_obj['current_ymin'] or cmax > line_graph_obj['current_ymax']:
                        ax_line.set_ylim(cmin, cmax)
                        line_graph_obj['current_ymin'] = cmin
                        line_graph_obj['current_ymax'] = cmax

                # Update trial max line if applicable.
                if "max_" in selected_metric:
                    if current_val > line_graph_obj['current_trial_max']:
                        line_graph_obj['current_trial_max'] = current_val
                        if not hasattr(line_graph_obj['line_trial_max'], 'xdata'):
                            line_graph_obj['line_trial_max'].set_data([frame_num], [current_val])
                        else:
                            oldx = line_graph_obj['line_trial_max'].get_xdata()
                            oldy = line_graph_obj['line_trial_max'].get_ydata()
                            new_x = np.append(oldx, frame_num)
                            new_y = np.append(oldy, current_val)
                            line_graph_obj['line_trial_max'].set_data(new_x, new_y)
                        line_graph_obj['line_trial_max'].set_visible(True)
                else:
                    line_graph_obj['line_trial_max'].set_visible(False)

                if trial_release is not None and frame >= release_frame:
                    if "release_" in selected_metric and 'trial_release_line' in line_graph_obj:
                        line_graph_obj['trial_release_line'].set_visible(True)

                if line_ongoing_text:
                    line_ongoing_text.set_text(one_liner_str)

            fig.canvas.draw_idle()

        # Build the animation.
        frames_iter = frames_to_animate if frames_to_animate else range(len(df))
        # Note: You might experiment with blit=True if your artists support it.
        anim = FuncAnimation(fig, update_func, frames=frames_iter, interval=1000/30, blit=False)
        inline_html = anim.to_jshtml() or "<p>Error: No animation output generated.</p>"
        wrapped_html = f"""<div style="width: 100%; margin: 0; padding: 0;">{inline_html}</div>"""

        if save_path is not None:
            if save_path.endswith(".html"):
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(wrapped_html)
            elif save_path.endswith(".gif"):
                anim.save(save_path, writer='pillow')
            elif save_path.endswith(".mp4"):
                anim.save(save_path, writer='ffmpeg')

        if notebook_mode:
            return HTML(wrapped_html)
        else:
            return wrapped_html

    except Exception as e:
        logger.error(f"Error in animate_trial_with_calc_bayes_shap_angle_meter: {e}")
        raise









# %matplotlib notebook




    
def display_combined_output(feedback_table, animation_html, feedback_mode):
    """
    Displays the feedback table and animation side by side.

    Parameters:
        feedback_table (pd.DataFrame): The feedback metrics table.
        animation_html (IPython.display.HTML): The animation HTML object.
        feedback_mode (str): The current feedback mode for labeling purposes.
    """
    # Convert Feedback Table to HTML
    feedback_table_html = feedback_table.to_html(index=False)

    # Create an HTML layout to display both the table and the animation side by side
    combined_html = f"""
    <div style="display: flex; justify-content: space-between;">
        <div style="width: 50%;">
            <h3>Feedback Table ({feedback_mode.capitalize()} Mode)</h3>
            {feedback_table_html}
        </div>
        <div style="width: 45%;">
            <h3>Animation</h3>
            {animation_html.data}
        </div>
    </div>
    """

    # Display the combined layout
    display(HTML(combined_html))


def run_shot_meter_animation(
    bayesian_metrics_json_path: str,
    merged_data_path: str,
    trial_id: str,
    selected_metric: str,
    feedback_mode: str = "bayesian",
    viewpoint_name: str = "diagonal_player_centric",
    connections: list = [
        ("R_EYE", "L_EYE"), ("R_EYE", "NOSE"), ("L_EYE", "NOSE"),
        ("R_EYE", "R_EAR"), ("L_EYE", "L_EAR"), ("R_SHOULDER", "L_SHOULDER"),
        ("R_SHOULDER", "R_ELBOW"), ("L_SHOULDER", "L_ELBOW"), ("R_ELBOW", "R_WRIST"),
        ("L_ELBOW", "L_WRIST"), ("R_SHOULDER", "R_HIP"), ("L_SHOULDER", "L_HIP"),
        ("R_HIP", "L_HIP"), ("R_HIP", "R_KNEE"), ("L_HIP", "L_KNEE"),
        ("R_KNEE", "R_ANKLE"), ("L_KNEE", "L_ANKLE"), ("R_WRIST", "R_1STFINGER"),
        ("R_WRIST", "R_5THFINGER"), ("L_WRIST", "L_1STFINGER"), ("L_WRIST", "L_5THFINGER"),
        ("R_ANKLE", "R_1STTOE"), ("R_ANKLE", "R_5THTOE"), ("L_ANKLE", "L_1STTOE"),
        ("L_ANKLE", "L_5THTOE"), ("R_ANKLE", "R_CALC"), ("L_ANKLE", "L_CALC"),
        ("R_1STTOE", "R_5THTOE"), ("L_1STTOE", "L_5THTOE"), ("R_1STTOE", "R_CALC"),
        ("L_1STTOE", "L_CALC"), ("R_5THTOE", "R_CALC"), ("L_5THTOE", "L_CALC"),
        ("R_1STFINGER", "R_5THFINGER"), ("L_1STFINGER", "L_5THFINGER")
    ],
    debug: bool = False,
    frames_to_animate: list = None,
    show_selected_metric: bool = False,
    polar_plot: bool = True,
    bar_plot: bool = True,
    line_plot: bool = True,
    bayesian_range_percentile: int = 10,
    calculated_range_percentile: int = 10,
    shap_range_percentile: int = 10,
    update_percentiles: bool = False,
    config: Optional["AppConfig"] = None,
    separate_display: bool = True,
    save_path: Optional[str] = None,
    # NEW: Pass notebook_mode so the pipeline can call the animation function appropriately.
    notebook_mode: bool = False,
    streamlit_app_paths: bool = False
):
    import os
    try:
        if (bayesian_range_percentile != 10 or calculated_range_percentile != 10 or shap_range_percentile != 10):
            update_percentiles = True
            if debug:
                print("Percentile values changed. Triggering update.")
        if streamlit_app_paths:
            bayesian_metrics_file_path="data/predictions/bayesian_optimization_results/bayesian_optimization_results.csv"
            final_ml_file_path="data/processed/final_ml_dataset.csv"
            final_ml_with_predictions_path="data/predictions/shap_results/final_predictions_with_shap.csv"
            pickle_path="data/preprocessor/features_info/final_ml_df_selected_features_columns.pkl"
        else:
            bayesian_metrics_file_path="../../data/predictions/bayesian_optimization_results/bayesian_optimization_results.csv"
            final_ml_file_path="../../data/processed/final_ml_dataset.csv"
            final_ml_with_predictions_path="../../data/predictions/shap_results/final_predictions_with_shap.csv"
            pickle_path="../../data/preprocessor/features_info/final_ml_df_selected_features_columns.pkl"
            
        if update_percentiles:
            if config is None:
                raise ValueError("A valid AppConfig must be provided when updating percentiles.")
            merged_data, bayesian_metrics_dict, classification_summary = automated_bayes_shap_summary(
                granular_data_path=merged_data_path,
                release_angles_output_dir=os.path.dirname(merged_data_path),
                max_angles_output_dir=os.path.dirname(merged_data_path),
                bayesian_metrics_file_path=bayesian_metrics_file_path,
                final_ml_file_path=final_ml_file_path,
                final_ml_with_predictions_path=final_ml_with_predictions_path,
                pickle_path=pickle_path,
                y_variable="result",
                bayes_min_max_range_percentile=bayesian_range_percentile,
                calc_feedback_range_percentile=calculated_range_percentile,
                output_dir=os.path.dirname(merged_data_path),
                output_filename="bayesian_shot_meter_granular_dataset.csv",
                debug=debug,
                reload_shap_data=True,
                new_shap_min_max_percentile=shap_range_percentile,
                config=config,
                streamlit_app_paths=streamlit_app_paths
            )
        else:
            merged_data = pd.read_csv(merged_data_path)
            bayesian_metrics_dict = load_bayesian_metrics_dict(json_path=bayesian_metrics_json_path, debug=debug)

        trial_data = merged_data[merged_data['trial_id'] == trial_id].sort_values(by='frame_time').reset_index(drop=True)
        if trial_data.empty:
            raise ValueError(f"No data found for trial_id {trial_id}.")
        # After processing merged data and trial data:
        if debug:
            print(f"[run_shot_meter_animation] Trial data for {trial_id} contains {len(trial_data)} frames (shape {trial_data.shape}).")
        
        metric_info = bayesian_metrics_dict.get(selected_metric.lower())
        if not metric_info:
            raise ValueError(f"Selected metric '{selected_metric}' not found in bayesian_metrics_dict.")
        selected_metric_filter_name = metric_info.get('filter_name', selected_metric)
        if debug:
            print(f"[run_shot_meter_animation] Selected metric filter name: {selected_metric_filter_name}")
        
        release_frames = trial_data.index[trial_data["release_point_filter"] == 1].tolist()
        release_frame = release_frames[0] if release_frames else None
        if debug:
            print(f"[run_shot_meter_animation] Release frame for trial {trial_id} is {release_frame}.")
        
        shooting_frames = trial_data[trial_data['shooting_motion'] == 1].index.tolist()
        if release_frame is not None and release_frame not in shooting_frames:
            shooting_frames.append(release_frame)
            if debug:
                print(f"[run_shot_meter_animation] Added release_frame {release_frame} to shooting_frames.")
        if debug:
            print(f"[run_shot_meter_animation] Animating {len(shooting_frames)} frames.")
        
        feedback_table = generate_feedback_table_all_metrics(feedback_mode, bayesian_metrics_dict, trial_data, debug=debug)
        # Pass notebook_mode to the animation function
        animation_html = animate_trial_with_calc_bayes_shap_angle_meter(
            df=trial_data,
            release_frame=release_frame,
            selected_metric=selected_metric,
            bayesian_metrics_dict=bayesian_metrics_dict,
            feedback_mode=feedback_mode,
            viewpoint_name=viewpoint_name,
            connections=connections,
            debug=debug,
            frames_to_animate=shooting_frames,
            show_selected_metric=show_selected_metric,
            polar_plot=polar_plot,
            bar_plot=bar_plot,
            line_plot=line_plot,
            save_path=save_path,
            notebook_mode=notebook_mode
        )
        
        if separate_display:
            display_separate_outputs(feedback_table, animation_html, feedback_mode)
        else:
            display_combined_output(feedback_table, animation_html, feedback_mode)
            
        if not debug:
            print("Shot meter animation completed successfully.")
        
        # Return the animation output (HTML object in notebook mode or raw string in Streamlit)
        return animation_html, feedback_table
    
    except Exception as e:
        print(f"[run_shot_meter_animation] Error: {e}")
        raise


def initialize_bar_meter(ax_bar, min_val, max_val):
    """
    Initialize a horizontal bar for the bar meter.
    """
    ax_bar.set_xlim([min_val, max_val])
    ax_bar.set_ylim([-0.5, 0.5])
    ax_bar.set_xlabel("Angle (degrees)")
    ax_bar.set_yticks([])
    bar_container = ax_bar.barh(y=0, width=0, height=0.4, color='red', align='center')
    return bar_container

def update_bar_meter(bar_container, current_angle, min_val=0, max_val=180):
    """
    Update the bar meter width based on the current angle.
    """
    bar = bar_container[0]
    clipped_angle = max(min_val, min(current_angle, max_val))
    bar.set_width(clipped_angle)

def initialize_line_graph(ax_line, static_min: float, static_max: float,
                          selected_metric_filter_name: str, selected_metric: str,
                          feedback_mode: str, debug=False):
    """
    Initialize the line graph subplot with dynamic metric tracking.
    static_min and static_max are used for the static min/max lines.
    """
    try:
        ax_line.set_title(f"Metric Over Time: {selected_metric}", fontsize=14, pad=30)
        ax_line.set_xlabel("Frame", fontsize=12)
        ax_line.set_ylabel(f"{selected_metric} (degrees)", fontsize=12)

        line_metric, = ax_line.plot([], [], lw=2, color='red',
                                    label=f"{selected_metric_filter_name} Ongoing")
        if feedback_mode.lower() == "shap":
            min_label = f"{selected_metric} SHAP Min"
            max_label = f"{selected_metric} SHAP Max"
        elif feedback_mode.lower() == "calculated":
            min_label = f"{selected_metric} Calc Min"
            max_label = f"{selected_metric} Calc Max"
        else:
            min_label = f"{selected_metric} Bayes Min"
            max_label = f"{selected_metric} Bayes Max"

        line_min = ax_line.axhline(y=static_min, color='blue', linestyle='--', label=min_label)
        line_max = ax_line.axhline(y=static_max, color='green', linestyle='--', label=max_label)

        line_trial_max, = ax_line.plot([], [], lw=2, color='red', linestyle='--',
                                       label=f"{selected_metric} Selected")
        line_trial_max.set_visible(False)

        data_frames = []
        data_values = []

        initial_lower = static_min - 5
        initial_upper = static_max + 5
        ax_line.set_ylim(initial_lower, initial_upper)
        ax_line.legend(loc='upper right')

        if debug:
            logger.debug("Line graph initialized with dynamic metric tracking.")

        return {
            'line_metric': line_metric,
            'line_min': line_min,
            'line_max': line_max,
            'line_trial_max': line_trial_max,
            'data_frames': data_frames,
            'data_values': data_values,
            'current_ymin': initial_lower,
            'current_ymax': initial_upper,
            'current_trial_max': static_max,
            'trial_max_reached': False
        }
    except Exception as e:
        logger.error(f"Error initializing line graph: {e}")
        raise


# Main execution
if __name__ == "__main__":


    # # Feedback mode = shap: use shap_{selected_metric}_min, shap_{selected_metric}_max, and the {selected_metric} for lines on the graph. 
    # # Feedback mode = bayesian: use {selected_metric}_bayes_min, {selected_metric}_bayes_max, and the {selected_metric} for lines on the graph. 
    # # Feedback mode = calculated: use {selected_metric}_optimal_min, {selected_metric}_optimal_max, and the {selected_metric} for lines on the graph. 

    # # shap feedback mode:
    # #     shap_{selected_metric}_unit_change
    # #     shap_{selected_metric}_unit
    # #     shap_{selected_metric}_direction
    # #     shap_{selected_metric}_importance
    # #     shap_{selected_metric}_goal
    # #     shap_{selected_metric}_min
    # #     shap_{selected_metric}_max
    # #     shap_{selected_metric}_classification
    # # calculated feedback mode:
    # #     {selected_metric}_filtered_optimal_min
    # #     {selected_metric}_filtered_optimal_max
    # #     {selected_metric}_shot_classification
    # # bayesian feedback mode:
    # #     {selected_metric}_bayes_min
    # #     {selected_metric}_bayes_max
    # #     {selected_metric}_bayes_optimized
    # #     {selected_metric}_bayes_classification
    # #     {selected_metric}__bayes_unit_change  
    # # **Select Feedback Mode**
    # feedback_mode = "shap"  # Options: 'bayesian', 'shap', 'calculated'



    
    # # **Display the Feedback Table and Animation Together**
    # display_combined_output(feedback_table, animation_html, feedback_mode)
    config_path = Path('../../data/model/preprocessor_config/preprocessor_config.yaml')
    try:
        config: AppConfig = load_config(config_path)
        print(f"Configuration loaded successfully from {config_path}.")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        exit(1)
        
    bayesian_metrics_json_path = "../../data/model/shot_meter_docs/bayesian_metrics_dict.json"
    merged_data_path = "../../data/processed/final_granular_dataset.csv"
    selected_trial_id = "T0001"
    selected_metric = "elbow_max_angle"
    
    # Define connections for the player skeleton (example list)

    
    _ , feedback_table = run_shot_meter_animation(
        bayesian_metrics_json_path=bayesian_metrics_json_path,
        merged_data_path=merged_data_path,  # changed here
        trial_id="T0001",
        selected_metric="elbow_max_angle",
        feedback_mode="shap", # bayesian, calculated, shap
        viewpoint_name="diagonal_player_centric",
        debug=True,
        polar_plot=True,
        bar_plot=True,
        line_plot=True,
        bayesian_range_percentile=10,
        calculated_range_percentile=10,
        shap_range_percentile=10,
        update_percentiles=False,
        save_path=None,
        config=config,
        streamlit_app_paths=False,
        notebook_mode=True,
        show_selected_metric=True
    )
    print("Feedback TABLE ===============", feedback_table)
