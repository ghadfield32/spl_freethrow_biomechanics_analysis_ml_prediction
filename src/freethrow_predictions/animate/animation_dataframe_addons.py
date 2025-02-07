# %%writefile ../../src/animation_dataframe_addons.py

import logging
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from IPython.display import HTML
from mplbasketball.court3d import Court3D, draw_court_3d

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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

# Define the connections between joints
CONNECTIONS = [
    # Skeletal connections for visualization
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
]

def initialize_plot(viewpoint_name: str, zlim: float, figsize=(10, 8), debug: bool = False) -> (plt.Figure, plt.Axes):
    """
    Initialize a 3D plot with specified viewpoint settings and outputs setup details.

    Parameters:
    - viewpoint_name (str): Name of the predefined viewpoint.
    - zlim (float): The limit for the z-axis (height).
    - figsize (tuple): Size of the figure.
    - debug (bool): Flag to enable debug logging.

    Returns:
    - fig: The Matplotlib figure object.
    - ax: The Matplotlib 3D axis object.
    """
    try:
        # Retrieve elev and azim from viewpoint name
        viewpoint = get_viewpoint(viewpoint_name)
        elev = viewpoint["elev"]
        azim = viewpoint["azim"]

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

def get_hoop_position(court_type: str = "nba", units: str = "ft", debug: bool = False) -> (float, float, float):
    """
    Calculate the 3D position of the basketball hoop based on court specifications.

    Parameters:
    - court_type (str): Type of the court ('nba', 'wnba', 'ncaa').
    - units (str): Units of measurement ('ft' or 'm').
    - debug (bool): Flag to enable debug logging.

    Returns:
    - x, y, z (float): Coordinates of the hoop in 3D space.
    """
    try:
        court = Court3D(court_type=court_type, units=units)
        params = court.court_parameters
        # The hoop is located a certain distance from the edge of the court
        x = params['court_dims'][0] / 2 - params['hoop_distance_from_edge']
        y = 0.0  # Centered along the y-axis
        z = params['hoop_height']
        if debug:
            logger.debug(f"Calculated hoop position at (x={x}, y={y}, z={z}) for court type '{court_type}' in '{units}' units.")
        return x, y, z
    except KeyError as e:
        logger.error(f"Key error when accessing court parameters: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_hoop_position: {e}")
        raise

def draw_court(ax: plt.Axes, court_type: str = "nba", units: str = "ft", debug: bool = False) -> None:
    """
    Draw the basketball court and hoop on the given axes.

    Parameters:
    - ax (plt.Axes): The Matplotlib 3D axis object.
    - court_type (str): Type of the court ('nba', 'wnba', 'ncaa').
    - units (str): Units of measurement ('ft' or 'm').
    - debug (bool): Flag to enable debug logging.

    Returns:
    - None
    """
    try:
        # Draw the court using mplbasketball
        draw_court_3d(ax, court_type=court_type, units=units, origin=np.array([0.0, 0.0]), line_width=2)
        if debug:
            logger.debug("Court drawn successfully.")

        # Get court parameters
        court = Court3D(court_type=court_type, units=units)
        court_params = court.court_parameters
        if debug:
            logger.debug(f"Court Parameters in draw_court: {court_params}")

        # Get hoop position
        hoop_x, hoop_y, hoop_z = get_hoop_position(court_type=court_type, units=units, debug=debug)

        # Draw the hoop as a circle
        hoop_radius = court_params['hoop_diameter'] / 2
        theta_circle = np.linspace(0, 2 * np.pi, 100)
        hoop_xs = hoop_x + hoop_radius * np.cos(theta_circle)
        hoop_ys = hoop_y + hoop_radius * np.sin(theta_circle)
        hoop_zs = np.full_like(hoop_xs, hoop_z)

        ax.plot(hoop_xs, hoop_ys, hoop_zs, c='orange', lw=3)
        if debug:
            logger.debug(f"Hoop drawn at position ({hoop_x}, {hoop_y}, {hoop_z}) with radius {hoop_radius}.")

        # Plot half-court line
        half_court_x = np.linspace(-court_params['court_dims'][0]/2, court_params['court_dims'][0]/2, 100)
        half_court_y = np.full_like(half_court_x, 0.0)
        half_court_z = np.full_like(half_court_x, 0.0)
        ax.plot(half_court_x, half_court_y, half_court_z, c='black', lw=2, linestyle='--')

        # Plot three-point arc
        three_point_arc_angle = np.deg2rad(court_params['three_point_arc_angle'])  # Convert to radians
        theta_arc = np.linspace(-three_point_arc_angle, three_point_arc_angle, 150)  # -75° to +75°
        three_point_radius = court_params['three_point_arc_diameter'] / 2  # 23.75 ft
        three_point_x = hoop_x + three_point_radius * np.cos(theta_arc)
        three_point_y = hoop_y + three_point_radius * np.sin(theta_arc)
        three_point_z = np.full_like(three_point_x, 0.0)
        ax.plot(three_point_x, three_point_y, three_point_z, c='purple', lw=2)

        # Plot straight lines (corners) of the three-point line
        # Calculate the end points of the arc at theta = ±75 degrees
        end_x_positive = hoop_x + three_point_radius * np.cos(three_point_arc_angle)
        end_y_positive = hoop_y + three_point_radius * np.sin(three_point_arc_angle)
        end_x_negative = hoop_x + three_point_radius * np.cos(-three_point_arc_angle)
        end_y_negative = hoop_y + three_point_radius * np.sin(-three_point_arc_angle)

        # Baseline x position
        baseline_x = court_params['court_dims'][0]/2  # 47 ft

        # Straight lines from arc end to sideline (approx. 1.25 ft)
        straight_line_y_positive = np.linspace(end_y_positive, court_params['court_dims'][1]/2, 50)
        straight_line_x_positive = np.full_like(straight_line_y_positive, baseline_x)
        straight_line_z_positive = np.full_like(straight_line_y_positive, 0.0)
        ax.plot(straight_line_x_positive, straight_line_y_positive, straight_line_z_positive, c='purple', lw=2)

        straight_line_y_negative = np.linspace(end_y_negative, -court_params['court_dims'][1]/2, 50)
        straight_line_x_negative = np.full_like(straight_line_y_negative, baseline_x)
        straight_line_z_negative = np.full_like(straight_line_y_negative, 0.0)
        ax.plot(straight_line_x_negative, straight_line_y_negative, straight_line_z_negative, c='purple', lw=2)

        # Plot sidelines
        sideline_x = np.linspace(-court_params['court_dims'][0]/2, court_params['court_dims'][0]/2, 100)
        sideline_y_positive = np.full_like(sideline_x, court_params['court_dims'][1]/2)
        sideline_z = np.full_like(sideline_x, 0.0)
        ax.plot(sideline_x, sideline_y_positive, sideline_z, c='blue', lw=2)

        sideline_y_negative = np.full_like(sideline_x, -court_params['court_dims'][1]/2)
        ax.plot(sideline_x, sideline_y_negative, sideline_z, c='blue', lw=2)

        # Plot baselines
        baseline_x_positive = np.full_like(court_params['court_dims'][1]/2, court_params['court_dims'][0]/2)
        baseline_y = np.linspace(-court_params['court_dims'][1]/2, court_params['court_dims'][1]/2, 100)
        baseline_z = np.full_like(baseline_y, 0.0)
        ax.plot(court_params['court_dims'][0]/2, baseline_y, baseline_z, c='green', lw=2)

        baseline_x_negative = np.full_like(court_params['court_dims'][1]/2, -court_params['court_dims'][0]/2)
        ax.plot(-court_params['court_dims'][0]/2, baseline_y, baseline_z, c='green', lw=2)

        if debug:
            logger.debug("Additional court features (half-court, three-point lines, sidelines, baselines) drawn successfully.")
    except Exception as e:
        logger.error(f"Error drawing court or hoop: {e}")
        raise


def initialize_elements(
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

    Returns:
    - lines (dict): Dictionary of line objects for each connection.
    - ball (plt.Line2D): The ball plot object.
    - release_text (plt.Text): Text object for release point indicator.
    - motion_text (plt.Text): Text object for motion phase indicator.
    - distance_text (plt.Text): Text object for distance to hoop.
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
    xbuffer: float,
    ybuffer: float,
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
    - xbuffer (float): Horizontal axis buffer (not used in fixed axes mode).
    - ybuffer (float): Vertical axis buffer (not used in fixed axes mode).
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
    
    
def animate_trial_from_df(
    df: pd.DataFrame,
    release_frame: int,
    viewpoint_name: str = "side_view_right",  # New parameter for viewpoint
    connections: list = CONNECTIONS,
    xbuffer: float = 4.0,
    ybuffer: float = 4.0,
    zlim: float = 15.0,
    player_color: str = "purple",
    player_lw: float = 2.0,
    ball_color: str = "#ee6730",
    ball_size: float = 20.0,
    highlight_color: str = "red",
    show_court: bool = True,
    court_type: str = "nba",
    units: str = "ft",
    notebook_mode: bool = True,
    debug: bool = False
) -> HTML:
    """
    Animate the trial from the provided DataFrame, showing the player's motion and the ball.

    Returns:
    - HTML: The animation in HTML format for notebook display or the animation object.
    """
    try:
        # Close any existing figures to prevent duplicate animations
        plt.close('all')

        if debug:
            logger.debug("Starting animation setup.")
            logger.debug(f"Total frames in DataFrame: {len(df)}")
            logger.debug(f"Release frame index provided: {release_frame}")
            logger.debug(f"Selected viewpoint: {viewpoint_name}")

        # Plot setup with predefined viewpoint
        fig, ax = initialize_plot(viewpoint_name, zlim, figsize=(12, 10), debug=debug)

        # Draw court and get hoop position
        if show_court:
            draw_court(ax, court_type=court_type, units=units, debug=debug)
            hoop_x, hoop_y, hoop_z = get_hoop_position(court_type=court_type, units=units, debug=debug)
        else:
            hoop_x, hoop_y, hoop_z = None, None, None

        # Initialize elements for animation (modified to capture distance_text)
        lines, ball, release_text, motion_text, distance_text = initialize_elements(
            ax, connections, player_color, player_lw, ball_color, ball_size, debug=debug
        )

        # Compute axes limits
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

        # Initialize min and max with player coordinates
        x_min = player_x.min() - xbuffer
        x_max = player_x.max() + xbuffer
        y_min = player_y.min() - ybuffer
        y_max = player_y.max() + ybuffer

        # Include hoop position in the limits if court is shown and hoop position is valid
        if show_court and hoop_x is not None and hoop_y is not None:
            x_min = min(x_min, hoop_x - xbuffer)
            x_max = max(x_max, hoop_x + xbuffer)
            y_min = min(y_min, hoop_y - ybuffer)
            y_max = max(y_max, hoop_y + ybuffer)

        if debug:
            logger.debug(f"Player X range: {player_x.min()} to {player_x.max()}")
            logger.debug(f"Player Y range: {player_y.min()} to {player_y.max()}")
            if show_court:
                logger.debug(f"Hoop position: ({hoop_x}, {hoop_y})")
            logger.debug(f"Using xbuffer: {xbuffer}, ybuffer: {ybuffer}")

        # Set fixed axes limits
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

        if debug:
            logger.debug(f"Set axes limits: x=({x_min}, {x_max}), y=({y_min}, {y_max})")

        # Set plot title to include the viewpoint name
        ax.set_title(f"Animation - {viewpoint_name}", fontsize=16)
        if debug:
            logger.debug(f"Set plot title to 'Animation - {viewpoint_name}'")

        # Create custom legend handles for static court features
        hoop_handle = Line2D([0], [0], color='orange', lw=3, label='Hoop')
        three_point_handle = Line2D([0], [0], color='purple', lw=2, label='Three-Point Line')
        half_court_handle = Line2D([0], [0], color='black', lw=2, linestyle='--', label='Half-Court Line')
        baseline_handle = Line2D([0], [0], color='blue', lw=2, label='Baseline')

        # Create handles for dynamic elements
        player_handle = Line2D([0], [0], color=player_color, lw=player_lw, label='Player')
        ball_handle = Line2D([0], [0], marker='o', color='w', label='Ball',
                             markerfacecolor=ball_color, markersize=10)

        # Add legend with both static and dynamic elements (excluding distance)
        ax.legend(handles=[hoop_handle, three_point_handle, half_court_handle, baseline_handle,
                           player_handle, ball_handle], loc='upper right')

        if debug:
            logger.debug("Legend added with static court features and dynamic elements.")

        # Update function for animation
        def update(frame: int):
            """
            Wrapper function for updating the frame in the animation.
            """
            update_frame(
                ax=ax,  # The 3D axis object
                frame=frame,  # Current frame number
                df=df,  # DataFrame containing motion data
                release_frame=release_frame,  # Frame index of the release point
                lines=lines,  # Dictionary of line objects for skeleton
                ball=ball,  # Ball object for animation
                release_text=release_text,  # Text object for release point
                motion_text=motion_text,  # Text object for motion phase
                connections=connections,  # Joint connections
                xbuffer=xbuffer,  # Horizontal axis buffer (not used here)
                ybuffer=ybuffer,  # Vertical axis buffer (not used here)
                ball_color=ball_color,  # Default ball color
                highlight_color=highlight_color,  # Highlight color for release point
                debug=debug  # Debugging flag
            )

            # Update the distance text
            if 'distance_to_hoop' in df.columns:
                distance = df.at[frame, 'distance_to_hoop']
                if not pd.isna(distance):
                    distance_text.set_text(f"Distance to Hoop: {distance:.2f} ft")
                else:
                    distance_text.set_text("")
            else:
                distance_text.set_text("")

        # Animation setup
        anim = FuncAnimation(fig, update, frames=len(df), interval=1000 / 30, blit=False)

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

