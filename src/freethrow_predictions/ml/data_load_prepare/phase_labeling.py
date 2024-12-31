
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def calculate_distance(x1, y1, z1, x2, y2, z2):
    """
    Calculate the Euclidean distance between two 3D points.

    Parameters:
    - x1, y1, z1: Coordinates of the first point.
    - x2, y2, z2: Coordinates of the second point.

    Returns:
    - The Euclidean distance as a float.
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def label_ball_in_hands(df, hand_threshold=0.4, debug=False):
    """
    Label when the ball is in the hands by calculating distances between the ball and finger positions.

    Parameters:
    - df: DataFrame containing ball and hand joint positions.
    - hand_threshold: Distance threshold to determine if the ball is in hand.
    - debug: If True, prints debug information.

    Returns:
    - df: DataFrame with 'ball_in_hands' column indicating if the ball is in hand (1) or not (0).
    """
    df['dist_ball_R_1STFINGER'] = calculate_distance(
        df['ball_x'], df['ball_y'], df['ball_z'],
        df['R_1STFINGER_x'], df['R_1STFINGER_y'], df['R_1STFINGER_z']
    )
    df['dist_ball_R_5THFINGER'] = calculate_distance(
        df['ball_x'], df['ball_y'], df['ball_z'],
        df['R_5THFINGER_x'], df['R_5THFINGER_y'], df['R_5THFINGER_z']
    )
    df['dist_ball_L_1STFINGER'] = calculate_distance(
        df['ball_x'], df['ball_y'], df['ball_z'],
        df['L_1STFINGER_x'], df['L_1STFINGER_y'], df['L_1STFINGER_z']
    )
    df['dist_ball_L_5THFINGER'] = calculate_distance(
        df['ball_x'], df['ball_y'], df['ball_z'],
        df['L_5THFINGER_x'], df['L_5THFINGER_y'], df['L_5THFINGER_z']
    )
    # Determine if ball is in hand based on threshold
    df['ball_in_hands'] = ((df['dist_ball_R_1STFINGER'] < hand_threshold) |
                           (df['dist_ball_R_5THFINGER'] < hand_threshold) |
                           (df['dist_ball_L_1STFINGER'] < hand_threshold) |
                           (df['dist_ball_L_5THFINGER'] < hand_threshold)).astype(int)
    
    if debug:
        print("Debug: Ball-in-hand labeling completed.")
        print("Debug: Ball-in-hand table columns =", df.columns)
        print("Debug: Ball-in-hand table additions example =", df[['dist_ball_L_5THFINGER', 'dist_ball_L_1STFINGER']].head(5))
    return df

def label_shooting_motion(df, debug=False):
    """
    Label frames indicating shooting motion based on ball position relative to average shoulder height,
    extending the motion for 5 frames after the ball leaves the hands.

    Parameters:
    - df: DataFrame with ball and shoulder data.
    - debug: If True, prints debug information.

    Returns:
    - df: DataFrame with 'shooting_motion' column (1 for shooting motion, 0 otherwise).
    """
    df = label_ball_in_hands(df, debug=debug)
    df['shooting_motion'] = 0
    df['avg_shoulder_height'] = (df['R_SHOULDER_z'] + df['L_SHOULDER_z']) / 2

    # Identify shooting motion conditionally
    start_motion_condition = (df['ball_in_hands'] == 1) & (df['ball_z'] >= df['avg_shoulder_height'])
    df.loc[start_motion_condition, 'shooting_motion'] = 1

    # Track when motion starts and propagate shooting motion
    df['motion_group'] = start_motion_condition.cumsum()  # Grouping identifier for shooting motion
    shooting_groups = df['motion_group'].unique()
    shooting_groups = shooting_groups[shooting_groups > 0]  # Ignore group 0 (non-shooting frames)

    # Extend shooting motion for 5 frames after the ball leaves the hands
    for group in shooting_groups:
        group_indices = df.index[df['motion_group'] == group].tolist()
        if group_indices:
            last_index = group_indices[-1]
            extension_indices = range(last_index + 1, last_index + 5)  # Extend by 5 frames
            valid_extension_indices = [i for i in extension_indices if i < len(df)]
            df.loc[valid_extension_indices, 'shooting_motion'] = 1

    # Drop the helper column if not needed for debugging
    df.drop(columns='motion_group', inplace=True)

    if debug:
        print("Debug: Shooting motion labeled and extended by 5 frames.")
        print("Debug: Shooting motion table columns =", df.columns)
        print("Debug: Shooting motion example =", df[['frame_time', 'shooting_motion', 'ball_in_hands']].head(15))
        
    return df



def find_release_point(df, debug=False):
    """
    Identify the release point of the shot by finding frames where elbows are above shoulder height 
    and then identifying the frame with the maximum vertical ball velocity during the shooting motion
    while the ball is still in hand.

    Parameters:
    - df: DataFrame containing motion and ball data.
    - debug: If True, prints step-by-step debug information.

    Returns:
    - df: DataFrame with 'release_point_filter' column marking the release point.
    """
    df['release_point_filter'] = 0  # Initialize with 0 for all frames

    # Make a copy of the subset to avoid modifying the original DataFrame directly
    shooting_motion_df = df[(df['shooting_motion'] == 1) & (df['ball_in_hands'] == 1)].copy()
    
    # Calculate the average shoulder height for frames within the shooting motion
    shooting_motion_df['avg_shoulder_height'] = (
        shooting_motion_df['R_SHOULDER_z'] + shooting_motion_df['L_SHOULDER_z']
    ) / 2
    
    if debug:
        print("Debug: Calculating average shoulder height for shooting frames.")
    
    # Filter frames where both elbows are above shoulder height
    elbows_above_shoulder_df = shooting_motion_df[
        (shooting_motion_df['R_ELBOW_z'] >= shooting_motion_df['avg_shoulder_height']) &
        (shooting_motion_df['L_ELBOW_z'] >= shooting_motion_df['avg_shoulder_height'])
    ]
    
    # Ensure we have enough frames after the threshold is met
    if not elbows_above_shoulder_df.empty:
        # Shift by 2 frames after the first occurrence of elbows above shoulder height
        first_above_shoulder_index = elbows_above_shoulder_df.index[0] + 3

        # Filter the subset to include frames starting from the two-frame offset
        filtered_df = shooting_motion_df.loc[first_above_shoulder_index:]
        
        # Identify the frame with maximum vertical ball velocity in the filtered set
        if not filtered_df.empty:
            max_velocity_frame = filtered_df.loc[
                filtered_df['ball_velocity_z'] == filtered_df['ball_velocity_z'].max(), 'frame_time'
            ].values
            
            if max_velocity_frame.size > 0:
                max_velocity_frame_time = max_velocity_frame[0]
                release_frame_index = filtered_df.index[filtered_df['frame_time'] == max_velocity_frame_time].tolist()
                
                if release_frame_index:
                    release_index = release_frame_index[0]
                    # Using .loc on the original df to avoid SettingWithCopyWarning
                    df.loc[release_index, 'release_point_filter'] = 1  # Mark release frame with 1
                    if debug:
                        print(f"Debug: Release frame found at index {release_index}, frame time {max_velocity_frame_time}")
                        print("Debug: Release point column updated for release frame.")
    else:
        if debug:
            print("Debug: No valid frames found for release point determination.")
    
    return df




def main_label_shot_phases(df, debug=False):
    """
    Main function to label phases of a basketball shot: identifies the shooting motion 
    and determines the release point.

    Parameters:
    - df: DataFrame containing ball and joint data.
    - debug: If True, prints debug information at each step.

    Returns:
    - df: DataFrame with labeled shot phases, including 'shooting_motion' and 'release_point_filter'.
    """
    df = label_shooting_motion(df, debug=debug)
    df = find_release_point(df, debug=debug)
    
    if debug:
        print("Debug: DataFrame head with shooting motion and release point labels:")
        print(df[['frame_time', 'shooting_motion', 'ball_in_hands', 'release_point_filter']].head(10))
    return df


# Example of how to use these functions in practice:
if __name__ == "__main__":
    import logging
    # from src.animation_dataframe_addons import animate_trial_from_df
    # from data_loading.load_and_parse import load_single_ft_and_parse
    # from data_preprocessing.dataframe_creation import main_create_dataframe
    # from velocity_and_speed_calc import calculate_ball_speed_velocity_direction
    # from shot_phase_labeling import main_label_shot_phases
    from IPython.display import display
    # from animate.animation import animate_trial_from_df

    # Default connections between joints
    connections = [
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

    # Configure logging for the script
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        # Load and prepare the DataFrame (replace with actual loading code)
        data, trial_id, result, landing_x, landing_y, entry_angle, _ = load_single_ft_and_parse(
            "../../SPL-Open-Data/basketball/freethrow/data/P0001/BB_FT_P0001_T0088.json",
            debug=False
        )
        df = main_create_dataframe(data, trial_id, result, landing_x, landing_y, entry_angle, debug=False)

        # Calculate ball speed, velocity, and direction
        df = calculate_ball_speed_velocity_direction(df, debug=False)

        # Label shot phases, find release frame, and mark the release point filter
        df = main_label_shot_phases(df, debug=False)

        print("Final DataFrame after labeling:", df)
        print("Final DataFrame columns:", df.columns)

        # Display animation if a release frame is marked in the DataFrame
        if 'release_point_filter' in df.columns and df['release_point_filter'].sum() > 0:
            release_frame_index = df.index[df['release_point_filter'] == 1].tolist()[0]

            # Set parameters for visualization
            viewpoint_name = "diagonal_player_centric"  # Choose from COMMON_VIEWPOINTS
            xbuffer = 10.0  # Adjust as needed
            ybuffer = 10.0  # Adjust as needed
            zlim = 15        # Adjust for height

            # Call the first animation function
            animation_html = animate_trial_from_df(
                df=df,
                release_frame=release_frame_index,
                viewpoint_name=viewpoint_name,
                connections=connections,
                zlim=zlim,
                player_color="purple",
                player_lw=2.0,
                ball_color="#ee6730",
                ball_size=20.0,
                highlight_color="red",
                show_court=True,
                court_type="nba",
                units="ft",
                notebook_mode=True,
                debug=True  # Enable detailed logging for troubleshooting
            )

            # Display the animation
            display(animation_html)
        else:
            print("Debug: No valid release frame found.")
    except Exception as e:
        logger.error(f"An error occurred in visualization: {e}")

