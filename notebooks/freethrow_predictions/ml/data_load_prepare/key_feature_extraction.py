import pandas as pd
import numpy as np
import json


def estimate_basketball_stats(height_m):
    """
    Estimate basketball-specific stats based on height.

    Args:
        height_m (float): Height of the player in meters.
    
    Returns:
        dict: Dictionary containing estimated basketball metrics.
    """
    # Convert height to centimeters for better readability in basketball context
    height_cm = height_m * 100
    
    # Estimate wingspan (average is roughly 1.06x height)
    wingspan_cm = height_cm * 1.06
    
    # Estimate standing reach (average for males is height in cm * 0.92 + 50 cm)
    standing_reach_cm = height_cm * 0.92 + 50
    
    # Estimate hand size (average male hand length is ~0.106x height in cm)
    hand_length_cm = height_cm * 0.106
    
    return {
        "player_estimated_wingspan_cm": round(wingspan_cm, 2),
        "player_estimated_standing_reach_cm": round(standing_reach_cm, 2),
        "player_estimated_hand_length_cm": round(hand_length_cm, 2)
    }

def load_player_info(json_path):
    """
    Load participant information from JSON file and add 'player_' prefix to each metric.
    Additionally, add a 'dominant_hand' key with the value 'R' to indicate right-handedness
    and calculate additional basketball-specific metrics based on height.

    Args:
        json_path (str): Path to the JSON file with player information.
    
    Returns:
        player_info (dict): Dictionary containing player-specific data with 'player_' prefix for each key.
    """
    with open(json_path, 'r') as f:
        player_info = json.load(f)
    
    # Add 'player_' prefix to each key
    player_info = {f"player_{key}": value for key, value in player_info.items()}
    
    # Add dominant hand key
    player_info["player_dominant_hand"] = "R"  # Setting dominant hand to 'R' (Right)
    
    # Estimate basketball-specific stats and add to player_info
    height_m = player_info.get("player_height_in_meters")
    if height_m:
        basketball_stats = estimate_basketball_stats(height_m)
        player_info.update(basketball_stats)
    
    return player_info



def get_column_definitions():
    """
    Define column descriptions for the dataset, including granular free throw data, ML features,
    energy metrics, exhaustion scores, and additional statistics.

    Returns:
        column_definitions (dict): Dictionary where keys are column names and values are descriptions.
    """
    column_definitions = {
        # Basic trial metadata
        'trial_id': "Unique identifier for each trial, formatted as 'Txxxx'.",
        'shot_id': "Sequential shot ID for organizing shots within a trial.",
        'result': "Binary indicator of shot outcome: 1 if made, 0 if missed.",
        
        # Free throw shot landing and entry characteristics
        'landing_x': "X coordinate of ball landing position on the hoop plane, measured in inches with the hoop front as origin.",
        'landing_y': "Y coordinate of ball landing position on the hoop plane, measured in inches with the hoop front as origin.",
        'entry_angle': "Angle at which the ball enters the hoop plane, measured in degrees to indicate entry precision.",

        # Frame and timing information
        'frame_time': "Timestamp in milliseconds for each frame, relative to trial start.",
        'dt': "Time delta between consecutive frames in seconds, used to calculate velocities and accelerations.",
        'by_trial_time': "Time within a specific trial, calculated as time elapsed from the trial start.",
        'continuous_frame_time': "Cumulative time across all trials, accounting for the trial offsets.",

        # Ball position and dynamics
        'ball_x': "X coordinate of the ball's position on the court, representing lateral position in feet, based on the center of the court.",
        'ball_y': "Y coordinate of the ball's position on the court, representing forward/backward position in feet from the center of the court.",
        'ball_z': "Z coordinate of the ball's position, representing height in feet off the court (vertical position).",
        'ball_speed': "Overall speed of the ball, derived from the 3D velocities (Pythagorean theorem of x, y, z velocities).",
        'overall_ball_velocity': "Magnitude of the ball's velocity vector, indicating its total speed at any moment.",

        # Exhaustion and energy metrics
        'joint_energy': "Energy expended by a specific joint in a single frame, calculated as power multiplied by time delta.",
        'joint_energy_by_trial': "Cumulative energy expended by a specific joint within a trial.",
        'joint_energy_by_trial_exhaustion_score': "Normalized exhaustion score for a specific joint within a trial, scaled to the maximum trial energy.",
        'joint_energy_overall_cumulative': "Cumulative energy expended by a specific joint across all trials.",
        'joint_energy_overall_exhaustion_score': "Normalized exhaustion score for a specific joint across all trials, scaled to the maximum overall energy.",

        'total_energy': "Total energy expended by all joints in a single frame, calculated as the sum of joint energies.",
        'by_trial_energy': "Cumulative energy expended during a specific trial, calculated as the sum of total energy across frames.",
        'by_trial_exhaustion_score': "Exhaustion score normalized within each trial, calculated as cumulative energy divided by maximum energy in the trial.",
        'overall_cumulative_energy': "Cumulative energy expended across all trials, calculated as a running sum of total energy across frames.",
        'overall_exhaustion_score': "Exhaustion score normalized across all trials, calculated as cumulative energy divided by the maximum cumulative energy.",

        # Joint-specific energy and exhaustion metrics
        'L_ANKLE_energy_by_trial': "Energy expended by the left ankle during a specific trial, calculated as power multiplied by time delta.",
        'L_ANKLE_energy_by_trial_exhaustion_score': "Exhaustion score for the left ankle energy in a trial, normalized to the maximum trial energy.",
        'L_ANKLE_energy_overall_cumulative': "Cumulative energy expended by the left ankle across all trials.",
        'L_ANKLE_energy_overall_exhaustion_score': "Exhaustion score for the left ankle energy across all trials, normalized to the maximum cumulative energy.",
        # Similarly for R_ANKLE, L_KNEE, R_KNEE, L_HIP, R_HIP, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST

        # Joint angles and release metrics
        'elbow_angle': "Angle at the elbow joint during motion, useful for tracking shooting form.",
        'wrist_angle': "Angle at the wrist joint, essential for understanding release dynamics.",
        'knee_angle': "Angle at the knee joint, analyzed to determine player’s stance and motion.",
        'max_elbow_angle': "Maximum angle achieved by the elbow during a trial, indicating the range of motion.",
        'max_wrist_angle': "Maximum angle achieved by the wrist during a trial, indicating the peak of wrist flexion or extension.",
        'max_knee_angle': "Maximum angle achieved by the knee during a trial, indicating the peak of knee flexion or extension.",
        'release_elbow_angle_filtered_optimal_min': "Minimum filtered optimal elbow angle at the release point during shooting motion.",
        'release_knee_angle_initial_optimal_max': "Maximum initial optimal knee angle during shooting motion at the release point.",
        'release_wrist_angle_initial_optimal_min': "Minimum initial optimal wrist angle during shooting motion at the release point.",

        # Power metrics
        'L_ANKLE_ongoing_power': "Calculated power (velocity) for the left ankle during motion.",
        'L_ELBOW_ongoing_power': "Calculated power (velocity) for the left elbow during motion.",
        'R_WRIST_ongoing_power': "Calculated power (velocity) for the right wrist during motion.",
        'L_1STFINGER_ongoing_power': "Calculated power (velocity) for the left first finger during motion.",
        'R_5THFINGER_ongoing_power': "Calculated power (velocity) for the right fifth finger during motion.",
        # Similarly for other joints

        # Shot classification and metrics
        'release_wrist_shot_classification': "Categorical classification of the shot based on wrist angle at the release point.",
        'max_knee_shot_classification': "Categorical classification of the shot based on the maximum knee angle during the trial.",
        'initial_release_angle': "Initial calculated release angle of the ball at the start of the shooting motion.",
        'optimal_release_angle': "Optimal release angle derived from shooting mechanics and trajectory analysis.",
        'calculated_release_angle': "Calculated release angle based on ball velocities and directions.",

        # Distances
        'dist_ball_R_1STFINGER': "Distance between the ball and the right first finger at each frame.",
        'dist_ball_L_5THFINGER': "Distance between the ball and the left fifth finger at each frame.",

        # Miscellaneous
        'avg_shoulder_height': "Average height of the shoulders during the trial, calculated as the mean of left and right shoulder heights.",
        'shooting_motion': "Binary indicator of whether the player is in a shooting motion (1) or not (0).",
        'release_point_filter': "Boolean filter indicating whether the frame meets release point criteria.",
        'initial_release_angle': "the release angle at the frame of the release_point_filter=1",
        'calculated_release_angle': "the average release angle over the 3 frames after the release_point_filter=1",
        'distance_to_basket': "the distance to the basket from the player, input manually by user. Can be calculated through YOLO",
        'optimal_release_angle': 'optimal_release_angle is calculated from the reference table from the 2005 study on optimal angle givent he release height, ball velocity, and such' ,
    }

    return column_definitions



def main_prepare_ml_data(df, joint_power_metrics_df, joint_angle_metrics_df, ml_metrics_df, player_info, debug=False):
    """
    Main function to prepare a single DataFrame for machine learning.

    Args:
        df: DataFrame containing the original data with shot details.
        joint_power_metrics_df: DataFrame with joint power metrics during the shooting motion.
        joint_angle_metrics_df: DataFrame with max and release angles during shooting motion.
        ml_metrics_df: DataFrame containing metrics specific to release dynamics and peak trajectory.
        player_info (dict): Dictionary containing player information (e.g., height, weight).
    
    Returns:
        key_features_dataframe: the final features of this free throw.
    """
    
    # Step 1: Extract essential shot details, including target variables and unique features
    shot_details_df = df[['result', 'landing_x', 'landing_y', 'entry_angle']].drop_duplicates().reset_index(drop=True)
    
    if debug:
        print("Debug: Shot details DataFrame:\n", shot_details_df)

    # Step 2: Combine all metrics and player info into a single row DataFrame for machine learning
    key_features_dataframe = pd.concat(
        [shot_details_df, joint_power_metrics_df, joint_angle_metrics_df, ml_metrics_df],
        axis=1
    )
    
    # Add player information as additional columns
    for key, value in player_info.items():
        key_features_dataframe[key] = value
    
    if debug:
        print("Debug: Combined key features DataFrame with player information for ML:\n", key_features_dataframe)

    return key_features_dataframe

if __name__ == "__main__":
    # Load and process data
    player_info_path = "../../SPL-Open-Data/basketball/freethrow/participant_information.json"
    player_info = load_player_info(player_info_path)
    
    print("Debug: Loaded player info:\n", player_info)
    
    data_path = "../../SPL-Open-Data/basketball/freethrow/data/P0001/BB_FT_P0001_T0001.json"
    print(f"Debug: Loading and parsing file: {data_path}")
    
    data, trial_id, result, landing_x, landing_y, entry_angle, _ = load_single_ft_and_parse(data_path)
    debug = True
    df = main_create_dataframe(data, trial_id, result, landing_x, landing_y, entry_angle, debug=False)
    
    df = calculate_ball_speed_velocity_direction(df, debug=False)
    df = main_label_shot_phases(df)
    release_frame_index = df.index[df['release_point_filter'] == 1].tolist()[0]
    
    df, release_index, projection_df, ml_metrics_df = main_ball_trajectory_analysis(df, release_frame_index, debug=True)
    print("Final DataFrame with release features and trajectory data:\n", df)
    print("ML Metrics DataFrame for model input:\n", ml_metrics_df)

    joints = [
        'L_ANKLE', 'R_ANKLE', 'L_KNEE', 'R_KNEE', 'L_HIP', 'R_HIP',
        'L_ELBOW', 'R_ELBOW', 'L_WRIST', 'R_WRIST',
        'L_1STFINGER', 'L_5THFINGER', 'R_1STFINGER', 'R_5THFINGER'
    ]

    # Step 1: Calculate joint power and metrics
    df, joint_power_metrics_df = main_calculate_joint_power(df, release_frame_index, debug=debug)
    
    # Now you have:
    # - df: Main DataFrame with added per-frame joint power columns
    # - joint_power_metrics_df: DataFrame with joint power metrics calculated during shooting motion
    
    print("DataFrame with ongoing joint power columns added:\n", df.head())
    print("Joint Power Metrics DataFrame (during shooting motion):\n", joint_power_metrics_df)

    # Calculate Joint Angles Across All Rows and Get Metrics for Shooting Motion
    df, joint_angle_metrics_df = calculate_joint_angles_over_motion(df, release_frame_index, side='R', debug=True)
    print("Final DataFrame with ongoing joint angles across all rows:\n", df.head())
    print("Joint Angle Metrics DataFrame (max and release angles during shooting motion):\n", joint_angle_metrics_df)

    # metrics df's for machine learning dataset and base dataset
    # - joint_power_metrics_df: DataFrame with joint power metrics calculated during shooting motion
    # - joint_angle_metrics_df (max and release angles during shooting motion)
    # - shot_details_df = df[['result', 'landing_x', 'landing_y', 'entry_angle']].drop_duplicates()
    # ^ shot_details_df are the base metrics from the dataset and the y variable = result

    # Prepare the ML dataset
    key_features_dataframe = main_prepare_ml_data(df, joint_power_metrics_df, joint_angle_metrics_df, ml_metrics_df, player_info, debug=True)
    print("Final Key Features DataFrame for ML:\n", key_features_dataframe)
    print("Final Key Features DataFrame Columns for ML:\n", key_features_dataframe.columns)

    #Final Two output tables: df (granular free throw data) and key_features_dataframe for the ML dataset
    
    # Column Definitions
    column_definitions = get_column_definitions()
    print("Column Definitions for the Dataset:\n", column_definitions)
