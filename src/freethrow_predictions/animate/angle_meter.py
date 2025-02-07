
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

logger = logging.getLogger(__name__)

def update_release_angle_line(ax_meter, release_angle, release_knee_shot_classification, release_angle_text, debug=False):
    """Draw the release knee angle line and update the release angle text with classification."""
    release_angle_rad = release_angle * np.pi / 180  # Convert to radians for polar plot
    ax_meter.plot([0, release_angle_rad], [0, 1], color='black', lw=2, linestyle="-")

    # Update the release angle text with classification
    release_angle_text.set_text(f'Release Knee Angle: {release_angle:.1f}° ({release_knee_shot_classification})')
    release_angle_text.set_zorder(15)  # Ensure it stays on top

    if debug:
        print(f"Debug: Release angle line added at {release_angle}° with classification '{release_knee_shot_classification}'")



def update_angle_meter(
    ax_meter,
    df,
    frame,
    needle,
    angle_text,
    max_angle_text,
    release_angle_text,
    max_angle,
    classification,
    release_angle,
    release_classification,
    angle_key,
    is_max_key,
    classification_key,
    release_classification_key,
    debug
):
    """Generalized function to update the angle meter for any joint."""
    try:
        current_angle = df.loc[frame, angle_key] * np.pi / 180  # Convert to radians
        needle.set_data([0, current_angle], [0, 1])  # Update needle position
        angle_text.set_text(f'Ongoing Angle: {df.loc[frame, angle_key]:.1f}°')

        if debug:
            logger.debug(f"Frame {frame}, Current angle for '{angle_key}': {df.loc[frame, angle_key]:.1f}°")

        if df.loc[frame, is_max_key] == 1 and max_angle is None:
            max_angle = df.loc[frame, angle_key]
            if classification_key in df.columns:
                classification = df.loc[frame, classification_key]
            else:
                classification = 'N/A'
                logger.warning(f"Column '{classification_key}' missing for joint '{angle_key}'.")
            update_max_angle_line(ax_meter, max_angle, classification, max_angle_text, debug=debug)

        if df.loc[frame, 'release_point_filter'] == 1 and release_angle is None:
            release_angle = df.loc[frame, angle_key]
            if release_classification_key in df.columns:
                release_classification = df.loc[frame, release_classification_key]
            else:
                release_classification = 'N/A'
                logger.warning(f"Column '{release_classification_key}' missing for joint '{angle_key}'.")
            update_release_angle_line(ax_meter, release_angle, release_classification, release_angle_text, debug=debug)

        return max_angle, classification, release_angle, release_classification
    except KeyError as e:
        logger.error(f"Missing column during angle meter update: {e}")
        raise

def update_max_angle_line(ax_meter, max_angle, classification, max_angle_text, debug=False):
    """
    Draw the maximum angle line and display max angle with classification.
    Display format: Max Knee Angle: {max_knee_angle}° ({max_knee_shot_classification})
    """
    try:
        max_angle_rad = max_angle * np.pi / 180  # Convert to radians for polar plot
        ax_meter.plot([0, max_angle_rad], [0, 1], color='black', lw=2, linestyle="-")

        # Update the max angle text content with zorder to ensure visibility
        max_angle_text.set_text(f'Max Angle: {max_angle:.1f}° ({classification})')
        max_angle_text.set_zorder(15)  # Ensure it stays on top

        if debug:
            logger.debug(f"Max angle line added at {max_angle}° with classification '{classification}' positioned near center.")
    except Exception as e:
        logger.error(f"Error in update_max_angle_line: {e}")
        raise

def add_optimal_lines(ax, min_angle, max_angle, release_min_angle, release_max_angle, debug=False):
    """
    Add optimal min and max lines to the angle meter, including release min and max angles.
    """
    try:
        # Convert angles to radians
        min_angle_rad = min_angle * np.pi / 180
        max_angle_rad = max_angle * np.pi / 180
        release_min_angle_rad = release_min_angle * np.pi / 180
        release_max_angle_rad = release_max_angle * np.pi / 180

        # Plotting optimal min and max lines
        ax.plot([0, min_angle_rad], [0, 1], color='darkblue', lw=2, linestyle="--")
        ax.plot([0, max_angle_rad], [0, 1], color='darkblue', lw=2, linestyle="--")

        # Plotting release min and max lines in red
        ax.plot([0, release_min_angle_rad], [0, 1], color='red', lw=2, linestyle="--")
        ax.plot([0, release_max_angle_rad], [0, 1], color='red', lw=2, linestyle="--")

        if debug:
            logger.debug(f"Added optimal lines at {min_angle}° and {max_angle}°, release lines at {release_min_angle}° and {release_max_angle}°")
    except Exception as e:
        logger.error(f"Error in add_optimal_lines: {e}")
        raise

def create_angle_meter(
    fig,
    selected_joint,
    min_angle,
    max_angle,
    release_min_angle,
    release_max_angle,
    text_offsets,
    debug=False
):
    """
    Create an angle meter plot for the selected joint.

    Parameters:
    - fig (plt.Figure): The figure to add the meter to.
    - selected_joint (str): The joint to display ('knee', 'elbow', 'wrist').
    - min_angle (float): The minimum optimal angle.
    - max_angle (float): The maximum optimal angle.
    - release_min_angle (float): The minimum release angle.
    - release_max_angle (float): The maximum release angle.
    - text_offsets (dict): Dictionary containing text offsets for the selected joint.
    - debug (bool): Flag to print debug information.

    Returns:
    - tuple: (ax_meter, needle, angle_text, max_angle_text, release_angle_text)
    """
    try:
        # Set up angle meter as a polar subplot for half-circle (180 degrees)
        ax_meter = fig.add_subplot(122, polar=True)
        ax_meter.set_theta_offset(np.pi)  # Rotate to make it horizontal
        ax_meter.set_theta_direction(-1)  # Counter-clockwise

        # Define 180-degree range with ticks and labels
        angle_ticks = np.linspace(0, np.pi, 6)
        angle_labels = [f'{int(angle)}°' for angle in np.linspace(0, 180, 6)]
        ax_meter.set_xticks(angle_ticks)
        ax_meter.set_xticklabels(angle_labels)
        ax_meter.set_yticklabels([])  # Hide radial labels for clarity

        # Limit the radial display to the upper half
        ax_meter.set_ylim(0, 1)  # Only show radial values from 0 to 1

        # Hide the lower half of the plot by masking the unwanted area
        ax_meter.set_facecolor("white")  # Set facecolor to blend with background
        ax_meter.fill_between(np.linspace(np.pi, 2 * np.pi, 100), 0, 1, color="white", zorder=10)  # Cover bottom half

        # Remove the full circular outline and add a 180-degree dividing line
        ax_meter.spines['polar'].set_visible(False)  # Hide full circular outline
        ax_meter.plot([0, np.pi], [0, 1], color='black', lw=1)  # Add a line along 180 degrees

        # Add optimal lines including release min and max lines
        add_optimal_lines(ax_meter, min_angle, max_angle, release_min_angle, release_max_angle, debug=debug)

        # Initialize the needle and text placeholders with adjustable y-positions
        needle, = ax_meter.plot([], [], color='r', lw=2)
        max_angle_text = ax_meter.text(
            0.5, text_offsets['max_angle_text_y'], '', transform=ax_meter.transAxes,
            ha='center', fontsize=14, color='black', zorder=15
        )
        angle_text = ax_meter.text(
            0.5, text_offsets['angle_text_y'], '', transform=ax_meter.transAxes,
            ha='center', fontsize=16, zorder=15
        )
        release_angle_text = ax_meter.text(
            0.5, text_offsets['release_angle_text_y'], '', transform=ax_meter.transAxes,
            ha='center', fontsize=14, color='black', zorder=15
        )

        if debug:
            logger.debug(f"Angle meter for '{selected_joint}' initialized with needle and angle text placeholders.")

        return ax_meter, needle, angle_text, max_angle_text, release_angle_text
    except Exception as e:
        logger.error(f"Error in create_angle_meter: {e}")
        raise


def initialize_angle_meter(
    fig: plt.Figure,
    df: pd.DataFrame,
    joint_config: dict,
    selected_joint: str,
    debug: bool = False
) -> dict:
    """
    Initialize the angle meter for the selected joint.

    Parameters:
    - fig (plt.Figure): The matplotlib figure object.
    - df (pd.DataFrame): DataFrame containing motion data.
    - joint_config (dict): Configuration dictionary for the joint.
    - selected_joint (str): The joint to display ('knee', 'elbow', 'wrist').
    - debug (bool): Flag to enable debug logging.

    Returns:
    - dict: Dictionary containing angle meter components.
    """
    angle_meter_obj = {}
    try:
        if joint_config['name'].lower() != selected_joint.lower():
            # Skip initialization for non-selected joints
            return angle_meter_obj

        # Define text offsets based on joint
        if selected_joint.lower() == 'knee':
            text_offsets = {
                'max_angle_text_y': 0.3,
                'angle_text_y': 0.35,
                'release_angle_text_y': 0.25
            }
        elif selected_joint.lower() == 'elbow':
            text_offsets = {
                'max_angle_text_y': 0.4,
                'angle_text_y': 0.45,
                'release_angle_text_y': 0.35
            }
        elif selected_joint.lower() == 'wrist':
            text_offsets = {
                'max_angle_text_y': 0.5,
                'angle_text_y': 0.55,
                'release_angle_text_y': 0.45
            }
        else:
            # Default offsets
            text_offsets = {
                'max_angle_text_y': 0.3,
                'angle_text_y': 0.35,
                'release_angle_text_y': 0.25
            }

        # Validate required columns
        required_columns = [
            joint_config['min_angle_key'],
            joint_config['max_angle_key'],
            joint_config['release_min_angle_key'],
            joint_config['release_max_angle_key']
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns for joint '{joint_config['name']}': {missing_columns}")
            raise KeyError(f"Missing required columns: {missing_columns}")

        min_angle = df[joint_config['min_angle_key']].iloc[0]
        max_angle = df[joint_config['max_angle_key']].iloc[0]
        release_min_angle = df[joint_config['release_min_angle_key']].iloc[0]
        release_max_angle = df[joint_config['release_max_angle_key']].iloc[0]

        # Create angle meter
        ax_meter, needle, angle_text, max_angle_text, release_angle_text = create_angle_meter(
            fig=fig,
            selected_joint=selected_joint,
            min_angle=min_angle,
            max_angle=max_angle,
            release_min_angle=release_min_angle,
            release_max_angle=release_max_angle,
            text_offsets=text_offsets,
            debug=debug
        )

        angle_meter_obj = {
            'ax_meter': ax_meter,
            'needle': needle,
            'angle_text': angle_text,
            'max_angle_text': max_angle_text,
            'release_angle_text': release_angle_text,
            'max_angle': None,
            'classification': None,
            'release_angle': None,
            'release_classification': None,
            'joint_config': joint_config  # Store joint_config for later use
        }

        if debug:
            logger.debug(f"Initialized angle meter for joint '{joint_config['name']}'.")

    except Exception as e:
        logger.error(f"Error initializing angle meter for joint '{joint_config['name']}': {e}")
        raise
    return angle_meter_obj

def initialize_selected_angle_meter(
    fig: plt.Figure,
    df: pd.DataFrame,
    joint_configs: list,
    selected_joint: str,
    debug: bool = False
) -> dict:
    """
    Initialize the angle meter for the selected joint among knee, elbow, and wrist.

    Parameters:
    - fig (plt.Figure): The matplotlib figure object.
    - df (pd.DataFrame): DataFrame containing motion data.
    - joint_configs (list): List of joint configuration dictionaries.
    - selected_joint (str): The joint to display ('knee', 'elbow', 'wrist').
    - debug (bool): Flag to enable debug logging.

    Returns:
    - dict: Dictionary containing angle meter components.
    """
    angle_meter_obj = {}
    try:
        for joint_config in joint_configs:
            if joint_config['name'].lower() == selected_joint.lower():
                angle_meter_obj = initialize_angle_meter(
                    fig=fig,
                    df=df,
                    joint_config=joint_config,
                    selected_joint=selected_joint,
                    debug=debug
                )
                break
        else:
            logger.error(f"Selected joint '{selected_joint}' is not recognized.")
            raise ValueError(f"Selected joint '{selected_joint}' is not valid. Choose from 'knee', 'elbow', 'wrist'.")
    except Exception as e:
        logger.error(f"Error initializing selected angle meter: {e}")
        raise
    return angle_meter_obj

def highlight_knee_joint(ax_3d, df, frame, knee_shade, debug=False):
    """
    Highlight the knee joint region from halfway up the upper leg to halfway down the lower leg.
    This function shades the back of the knee by creating a polygon between the midpoints of the upper leg and lower leg,
    updating dynamically as the knee angle changes.

    Parameters:
    - ax_3d: The 3D axis on which the highlight should be drawn.
    - df: DataFrame containing the joint coordinates.
    - frame: Current frame number in the animation.
    - knee_shade: The existing Poly3DCollection object for shading the knee, which will be updated.
    - debug: If True, print debug information.
    """
    try:
        # If knee_shade already exists, remove it from the current 3D axis
        if knee_shade is not None:
            knee_shade.remove()

        # Calculate the coordinates for the knee joint shading region
        r_knee_x, r_knee_y, r_knee_z = df.loc[frame, ["R_KNEE_x", "R_KNEE_y", "R_KNEE_z"]]
        r_ankle_x, r_ankle_y, r_ankle_z = df.loc[frame, ["R_ANKLE_x", "R_ANKLE_y", "R_ANKLE_z"]]
        r_hip_x, r_hip_y, r_hip_z = df.loc[frame, ["R_HIP_x", "R_HIP_y", "R_HIP_z"]]

        # Calculate halfway points for the upper leg and lower leg
        r_upper_mid_x, r_upper_mid_y, r_upper_mid_z = (r_knee_x + r_hip_x) / 2, (r_knee_y + r_hip_y) / 2, (r_knee_z + r_hip_z) / 2
        r_lower_mid_x, r_lower_mid_y, r_lower_mid_z = (r_knee_x + r_ankle_x) / 2, (r_knee_y + r_ankle_y) / 2, (r_knee_z + r_ankle_z) / 2

        # Define vertices for the shaded polygon at the back of the knee
        verts = [
            [r_upper_mid_x, r_upper_mid_y, r_upper_mid_z],
            [r_knee_x, r_knee_y, r_knee_z],
            [r_lower_mid_x, r_lower_mid_y, r_lower_mid_z]
        ]

        # Create a new Poly3DCollection for shading
        knee_shade = Poly3DCollection([verts], color='green', alpha=0.5)  # Set alpha for transparency
        ax_3d.add_collection3d(knee_shade)

        if debug:
            logger.debug(
                f"Frame {frame}, Knee joint shaded from upper midpoint ({r_upper_mid_x}, {r_upper_mid_y}, {r_upper_mid_z}) "
                f"to knee ({r_knee_x}, {r_knee_y}, {r_knee_z}) to lower midpoint ({r_lower_mid_x}, {r_lower_mid_y}, {r_lower_mid_z})"
            )

        return knee_shade  # Return the updated shading for the next frame
    except Exception as e:
        logger.error(f"Error in highlight_knee_joint: {e}")
        raise

def highlight_joint(
    ax: plt.Axes,
    df: pd.DataFrame,
    frame: int,
    joint_config: dict,
    joint_shade: Poly3DCollection = None,
    debug: bool = False
) -> Poly3DCollection:
    """
    Highlight the joint region dynamically as the angle changes.

    Parameters:
    - ax (plt.Axes): The 3D axis on which the highlight should be drawn.
    - df (pd.DataFrame): DataFrame containing the joint coordinates.
    - frame (int): Current frame number.
    - joint_config (dict): Configuration dictionary for the joint.
    - joint_shade (Poly3DCollection): Existing shaded region for the joint.
    - debug (bool): Flag to enable debug logging.

    Returns:
    - Poly3DCollection: Updated shaded region.
    """
    try:
        # Remove existing shade if present
        if joint_shade is not None:
            joint_shade.remove()

        # Example for knee joint; extend logic for other joints as needed
        joint = joint_config['name'].upper()  # e.g., 'knee' -> 'KNEE'
        joint_x = df.loc[frame, f"R_{joint}_x"]
        joint_y = df.loc[frame, f"R_{joint}_y"]
        joint_z = df.loc[frame, f"R_{joint}_z"]

        # Define vertices for shading (example logic)
        verts = [
            [joint_x - 0.5, joint_y - 0.5, joint_z],
            [joint_x + 0.5, joint_y - 0.5, joint_z],
            [joint_x + 0.5, joint_y + 0.5, joint_z],
            [joint_x - 0.5, joint_y + 0.5, joint_z]
        ]

        # Create a new Poly3DCollection for shading
        joint_shade = Poly3DCollection([verts], color='green', alpha=0.3)
        ax.add_collection3d(joint_shade)

        if debug:
            logger.debug(f"Frame {frame}, '{joint_config['name'].capitalize()}' joint shaded at ({joint_x}, {joint_y}, {joint_z}).")

        return joint_shade
    except Exception as e:
        logger.error(f"Error in highlight_joint for '{joint_config['name']}': {e}")
        raise

def initialize_angle_meters(
    fig: plt.Figure,
    df: pd.DataFrame,
    joint_configs: list,
    text_offsets: list,
    debug: bool = False
) -> dict:
    """
    Initialize angle meters for specified joints.

    Parameters:
    - fig (plt.Figure): The matplotlib figure object.
    - df (pd.DataFrame): DataFrame containing motion data.
    - joint_configs (list): List of joint configuration dictionaries.
    - text_offsets (list of lists): Each sublist contains [max_angle_text_y, angle_text_y, release_angle_text_y] for a joint.
    - debug (bool): Flag to enable debug logging.

    Returns:
    - dict: Dictionary containing angle meter components keyed by joint name.
    """
    angle_meter_objs = {}
    try:
        for idx, joint_config in enumerate(joint_configs):
            if idx < len(text_offsets):
                max_angle_text_y, angle_text_y, release_angle_text_y = text_offsets[idx]
            else:
                # Default offsets if not enough provided
                max_angle_text_y, angle_text_y, release_angle_text_y = 0.3, 0.35, 0.25

            # Validate required columns
            required_columns = [
                joint_config['min_angle_key'],
                joint_config['max_angle_key'],
                joint_config['release_min_angle_key'],
                joint_config['release_max_angle_key']
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns for joint '{joint_config['name']}': {missing_columns}")
                raise KeyError(f"Missing required columns: {missing_columns}")

            min_angle = df[joint_config['min_angle_key']].iloc[0]
            max_angle = df[joint_config['max_angle_key']].iloc[0]
            release_min_angle = df[joint_config['release_min_angle_key']].iloc[0]
            release_max_angle = df[joint_config['release_max_angle_key']].iloc[0]

            # Construct text_offsets as a dictionary
            text_offsets_dict = {
                'max_angle_text_y': max_angle_text_y,
                'angle_text_y': angle_text_y,
                'release_angle_text_y': release_angle_text_y
            }

            # Create angle meter
            ax_meter, needle, angle_text, max_angle_text, release_angle_text = create_angle_meter(
                fig=fig,
                selected_joint=joint_config['name'],
                min_angle=min_angle,
                max_angle=max_angle,
                release_min_angle=release_min_angle,
                release_max_angle=release_max_angle,
                text_offsets=text_offsets_dict,
                debug=debug
            )

            angle_meter_objs[joint_config['name']] = {
                'ax_meter': ax_meter,
                'needle': needle,
                'angle_text': angle_text,
                'max_angle_text': max_angle_text,
                'release_angle_text': release_angle_text,
                'max_angle': None,
                'classification': None,
                'release_angle': None,
                'release_classification': None,
                'joint_config': joint_config  # Store joint_config for later use
            }

            if debug:
                logger.debug(
                    f"Initialized angle meter for '{joint_config['name']}' with y-offsets: "
                    f"max_angle_text_y={max_angle_text_y}, angle_text_y={angle_text_y}, "
                    f"release_angle_text_y={release_angle_text_y}"
                )
    except Exception as e:
        logger.error(f"Error initializing angle meters: {e}")
        raise
    return angle_meter_objs




