
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from IPython.display import HTML
from mplbasketball.court3d import Court3D, draw_court_3d

from animate.elements import initialize_elements, initialize_plot
from animate.court import draw_court, get_hoop_position
from animate.viewpoints import get_viewpoint
from animate.angle_meter import (
    create_angle_meter,
    update_max_angle_line,
    update_release_angle_line,
    highlight_joint,
    highlight_knee_joint,
    update_angle_meter, 
    initialize_angle_meters
)

logger = logging.getLogger(__name__)

def update_frame(
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
    debug: bool = False
) -> None:
    """
    Update function for each frame in the animation.

    Parameters:
    - ax (plt.Axes): The Matplotlib 3D axis object.
    - frame (int): The current frame number.
    - df (pd.DataFrame): DataFrame containing motion data.
    - release_frame (int): Frame index of the release point.
    - lines (dict): Dictionary of line objects for skeleton.
    - ball (plt.Line2D): Ball object for animation.
    - release_text (plt.Text): Text object for release point.
    - motion_text (plt.Text): Text object for motion phase.
    - connections (list): Joint connections.
    - ball_color (str): Default ball color.
    - highlight_color (str): Highlight color for release point.
    - debug (bool): Flag to enable debug logging.

    Returns:
    - None
    """
    try:
        if debug and frame % 10 == 0:
            logger.debug(f"Updating frame {frame}")

        # Highlight the release frame
        if frame == release_frame:
            ball.set_color(highlight_color)
            release_text.set_text("Release Point!")
            if debug:
                logger.debug(f"Frame {frame} is the release frame. Ball color changed to {highlight_color}.")
        else:
            ball.set_color(ball_color)
            release_text.set_text("")

        # Update motion phase text if 'shooting_motion' exists
        if 'shooting_motion' in df.columns:
            shooting_motion = df.at[frame, 'shooting_motion']
            motion_text.set_text("Shooting Motion" if shooting_motion == 1 else "")
        else:
            motion_text.set_text("")

        # Update lines for joints
        for connection in connections:
            part1, part2 = connection
            if (f"{part1}_x" in df.columns and f"{part2}_x" in df.columns and
                not pd.isna(df.at[frame, f"{part1}_x"]) and not pd.isna(df.at[frame, f"{part2}_x"])):
                x = [df.at[frame, f"{part1}_x"], df.at[frame, f"{part2}_x"]]
                y = [df.at[frame, f"{part1}_y"], df.at[frame, f"{part2}_y"]]
                z = [df.at[frame, f"{part1}_z"], df.at[frame, f"{part2}_z"]]
                lines[connection].set_data_3d(x, y, z)
            else:
                # If data is missing, hide the line
                lines[connection].set_data([], [])
                lines[connection].set_3d_properties([])

        # Update ball position if ball coordinates exist
        if 'ball_x' in df.columns and 'ball_y' in df.columns and 'ball_z' in df.columns:
            ball_x = df.at[frame, 'ball_x']
            ball_y = df.at[frame, 'ball_y']
            ball_z = df.at[frame, 'ball_z']
            if not (pd.isna(ball_x) or pd.isna(ball_y) or pd.isna(ball_z)):
                ball.set_data_3d([ball_x], [ball_y], [ball_z])
            else:
                # Hide the ball if data is missing
                ball.set_data([], [])
                ball.set_3d_properties([])
        else:
            # Hide the ball if columns are missing
            ball.set_data([], [])
            ball.set_3d_properties([])
    except Exception as e:
        logger.error(f"Error updating frame {frame}: {e}")
        raise

def animate_trial_with_angle_meter(
    df: pd.DataFrame,
    release_frame: int,
    viewpoint_name: str = "side_view_right",
    connections: list = None,
    zlim: float = 15.0,
    joint_configs: list = None,  # Added parameter
    player_color: str = "purple",
    player_lw: float = 2.0,
    ball_color: str = "#ee6730",
    ball_size: float = 20.0,
    highlight_color: str = "red",
    show_court: bool = True,
    court_type: str = "nba",
    units: str = "ft",
    notebook_mode: bool = True,
    debug: bool = False,
    text_offsets: list = None  # Changed from individual y-offsets to a list of lists
) -> HTML:
    """
    Animate a basketball trial with an integrated angle meter.

    Parameters:
    - df (pd.DataFrame): DataFrame containing motion data.
    - release_frame (int): Frame index of the release point.
    - viewpoint_name (str): Name of the predefined viewpoint.
    - connections (list): List of joint connections.
    - zlim (float): The limit for the z-axis (height).
    - joint_configs (list): List of joint configuration dictionaries.
    - player_color (str): Color for player skeleton.
    - player_lw (float): Line width for player skeleton.
    - ball_color (str): Color for the ball.
    - ball_size (float): Size of the ball marker.
    - highlight_color (str): Highlight color for release point.
    - show_court (bool): Whether to display the court.
    - court_type (str): Type of the court ('nba', 'wnba', 'ncaa').
    - units (str): Units of measurement ('ft' or 'm').
    - notebook_mode (bool): Whether to display animation in Jupyter notebook.
    - debug (bool): Flag to enable debug logging.
    - text_offsets (list of lists): Each sublist contains [max_angle_text_y, angle_text_y, release_angle_text_y] for a joint.

    Returns:
    - HTML: HTML representation of the animation for notebook display.
    """
    try:
        if connections is None:
            logger.error("No connections provided for player skeleton.")
            raise ValueError("Connections list cannot be None.")
        
        if joint_configs is None:
            logger.error("No joint_configs provided for angle meters.")
            raise ValueError("joint_configs cannot be None.")
        
        if text_offsets is None:
            # Define default text_offsets if not provided
            text_offsets = [
                [0.3, 0.35, 0.25],  # For 'knee'
                [0.3, 0.35, 0.25],  # For 'elbow'
                [0.3, 0.35, 0.25]   # For 'wrist'
            ]

        # Close any existing figures to prevent duplicate animations
        plt.close('all')

        if debug:
            logger.debug("Starting animation setup.")
            logger.debug(f"Total frames in DataFrame: {len(df)}")
            logger.debug(f"Release frame index provided: {release_frame}")
            logger.debug(f"Selected viewpoint: {viewpoint_name}")

        # Retrieve elev and azim based on viewpoint_name
        try:
            viewpoint = get_viewpoint(viewpoint_name)
            elev = viewpoint['elev']
            azim = viewpoint['azim']
            if debug:
                logger.debug(f"Retrieved viewpoint '{viewpoint_name}': elev={elev}, azim={azim}")
        except KeyError:
            logger.error(f"Invalid viewpoint_name: {viewpoint_name}. Using default viewpoint.")
            viewpoint = get_viewpoint("side_view_right")  # Fallback to default
            elev = viewpoint['elev']
            azim = viewpoint['azim']

        # Plot setup with predefined viewpoint
        fig, ax = initialize_plot(zlim=zlim, elev=elev, azim=azim, figsize=(14, 8), debug=debug)

        # Draw court and get hoop position
        if show_court:
            draw_court(ax, court_type=court_type, units=units, debug=debug)
            hoop_x, hoop_y, hoop_z = get_hoop_position(court_type=court_type, units=units, debug=debug)
            if debug:
                logger.debug(f"Hoop position retrieved: ({hoop_x}, {hoop_y}, {hoop_z})")
        else:
            hoop_x, hoop_y, hoop_z = None, None, None
            if debug:
                logger.debug("Court not shown. Hoop position set to None.")

        # Initialize elements for animation
        lines, ball, release_text, motion_text, distance_text, joint_markers = initialize_elements(
            ax, connections, player_color, player_lw, ball_color, ball_size, debug=debug
        )

        # Initialize angle meters with optimal angle lines and adjustable text positions
        angle_meter_objs = initialize_angle_meters(
            fig=fig,
            df=df,
            joint_configs=joint_configs,  # Now correctly passed
            text_offsets=text_offsets,  # Pass list of lists
            debug=debug
        )
        if debug:
            logger.debug(f"Initialized {len(angle_meter_objs)} angle meters.")

        # Compute axes limits based on player data and hoop position
        if debug:
            logger.debug("Calculating axes limits to include player and hoop.")

        # Extract all x and y coordinates for the player
        player_x_cols = [col for col in df.columns if col.endswith('_x')]
        player_y_cols = [col for col in df.columns if col.endswith('_y')]

        player_x = df[player_x_cols].values.flatten()
        player_y = df[player_y_cols].values.flatten()

        # Remove NaN values
        player_x = player_x[~np.isnan(player_x)]
        player_y = player_y[~np.isnan(player_y)]

        if len(player_x) == 0 or len(player_y) == 0:
            logger.warning("No player coordinates found. Axes limits may not be set correctly.")

        # Get court parameters
        if show_court:
            court = Court3D(court_type=court_type, units=units)
            court_params = court.court_parameters
        else:
            court_params = {'court_length': 94.0, 'court_width': 50.0}  # Default values

        # Initialize min and max with player coordinates
        x_min = player_x.min() - 10.0 if len(player_x) > 0 else -court_params['court_length']/2
        x_max = player_x.max() + 10.0 if len(player_x) > 0 else court_params['court_length']/2
        y_min = player_y.min() - 10.0 if len(player_y) > 0 else -court_params['court_width']/2
        y_max = player_y.max() + 10.0 if len(player_y) > 0 else court_params['court_width']/2

        # Include hoop position in the limits if court is shown and hoop position is valid
        if show_court and hoop_x is not None and hoop_y is not None:
            x_min = min(x_min, hoop_x - 10.0)
            x_max = max(x_max, hoop_x + 10.0)
            y_min = min(y_min, hoop_y - 10.0)
            y_max = max(y_max, hoop_y + 10.0)
            if debug:
                logger.debug(f"Including hoop position in axes limits.")

        if debug:
            logger.debug(f"Player X range: {player_x.min()} to {player_x.max()}")
            logger.debug(f"Player Y range: {player_y.min()} to {player_y.max()}")
            if show_court:
                logger.debug(f"Hoop position: ({hoop_x}, {hoop_y})")
            logger.debug(f"Using xbuffer: 10.0, ybuffer: 10.0")

        # Set fixed axes limits
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

        if debug:
            logger.debug(f"Set axes limits: x=({x_min}, {x_max}), y=({y_min}, {y_max})")

        # Set plot title to include the viewpoint name
        ax.set_title(f"Animation with Angle Meter - {viewpoint_name}", fontsize=16)
        if debug:
            logger.debug(f"Set plot title to 'Animation with Angle Meter - {viewpoint_name}'")

        # Create custom legend handles for static court features
        hoop_handle = Line2D([0], [0], color='orange', lw=3, label='Hoop')
        baseline_handle = Line2D([0], [0], color='blue', lw=2, label='Baseline')
        sideline_handle = Line2D([0], [0], color='green', lw=2, label='Sideline')

        # Create handles for dynamic elements
        player_handle = Line2D([0], [0], color=player_color, lw=player_lw, label='Player')
        ball_handle = Line2D([0], [0], marker='o', color='w', label='Ball',
                             markerfacecolor=ball_color, markersize=10)

        # Add legend with both static and dynamic elements (excluding distance)
        ax.legend(handles=[
            hoop_handle,        # Hoop in orange
            sideline_handle,    # Sideline in green
            baseline_handle,    # Baseline in blue
            player_handle,      # Player skeleton in purple
            ball_handle         # Ball in orange
        ], loc='upper right')

        if debug:
            logger.debug("Legend added with static court features and dynamic elements.")

        # Define the update function for animation
        def update_func(frame: int):
            """
            Wrapper function for updating the frame in the animation.
            """
            # Update 3D elements
            update_frame(
                ax=ax,
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
                debug=debug
            )

            # Update angle meters for each joint
            for joint_name, meter in angle_meter_objs.items():
                # Extract joint-specific details
                joint_config = meter['joint_config']
                ax_meter = meter['ax_meter']
                needle = meter['needle']
                angle_text_meter = meter['angle_text']
                max_angle_text = meter['max_angle_text']
                release_angle_text = meter['release_angle_text']

                # Update angle meter
                meter['max_angle'], meter['classification'], \
                meter['release_angle'], meter['release_classification'] = update_angle_meter(
                    ax_meter=ax_meter,
                    df=df,
                    frame=frame,
                    needle=needle,
                    angle_text=angle_text_meter,
                    max_angle_text=max_angle_text,
                    release_angle_text=release_angle_text,
                    max_angle=meter['max_angle'],
                    classification=meter['classification'],
                    release_angle=meter['release_angle'],
                    release_classification=meter['release_classification'],
                    angle_key=joint_config['angle_key'],
                    is_max_key=joint_config['is_max_key'],
                    classification_key=joint_config['classification_key'],
                    release_classification_key=joint_config['release_classification_key'],
                    debug=debug
                )

            # Update the distance text if available
            if 'distance_to_hoop' in df.columns:
                distance = df.at[frame, 'distance_to_hoop']
                if not pd.isna(distance):
                    distance_text.set_text(f"Distance to Hoop: {distance:.2f} ft")
                else:
                    distance_text.set_text("")
            else:
                distance_text.set_text("")

        # Create and return animation
        anim = FuncAnimation(fig, update_func, frames=len(df), interval=1000 / 30, blit=False)

        if notebook_mode:
            if debug:
                logger.debug("Returning animation for notebook display.")
            return HTML(anim.to_jshtml())
        else:
            if debug:
                logger.debug("Returning animation object.")
            return anim
    except Exception as e:
        logger.error(f"An error occurred during animation: {e}")
        raise
