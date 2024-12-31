import numpy as np
import pandas as pd

def calculate_joint_angle(joint_a, joint_b, joint_c):
    """
    Calculate the angle between three points (joint_a, joint_b, joint_c).
    """
    vector_ab = joint_b - joint_a
    vector_bc = joint_c - joint_b
    dot_product = np.dot(vector_ab, vector_bc)
    mag_ab = np.linalg.norm(vector_ab)
    mag_bc = np.linalg.norm(vector_bc)

    if mag_ab == 0 or mag_bc == 0:
        return 0.0

    angle = np.arccos(dot_product / (mag_ab * mag_bc))
    return np.degrees(angle)

def calculate_joint_angles_over_motion(df, release_frame_index, side='R', debug=False):
    """
    Calculate joint angles over motion, allowing selection of side ('L' for left, 'R' for right).

    Parameters:
    - df: DataFrame containing motion data.
    - release_frame_index: Index for the release point.
    - side: 'L' for left side, 'R' for right side.
    - debug: If True, prints debug information.

    Returns:
    - df: Updated DataFrame with joint angle columns.
    - joint_angle_metrics_df: DataFrame with max and release angles for the specified joints.
    """
    # Set up joint combinations based on the selected side
    joint_combinations = {
        'elbow': [f'{side}_SHOULDER', f'{side}_ELBOW', f'{side}_WRIST'],
        'wrist': [f'{side}_ELBOW', f'{side}_WRIST', f'{side}_1STFINGER'],
        'knee': [f'{side}_HIP', f'{side}_KNEE', f'{side}_ANKLE']
    }

    # Step 1: Calculate angles for all rows in the DataFrame
    for joint_name, (a, b, c) in joint_combinations.items():
        df[f'{joint_name}_angle'] = df.apply(
            lambda row: calculate_joint_angle(
                row[[f"{a}_x", f"{a}_y", f"{a}_z"]].values,
                row[[f"{b}_x", f"{b}_y", f"{b}_z"]].values,
                row[[f"{c}_x", f"{c}_y", f"{c}_z"]].values
            ), axis=1
        )
        if debug:
            print(f"Debug: Calculated {side} {joint_name} angles across all rows:\n", df[[f'{joint_name}_angle']].head())

    # Step 2: Filter the DataFrame to get only the shooting motion rows
    shooting_motion_df = df[df['shooting_motion'] == 1]

    # Step 3: Initialize dictionary for max and release angles with single-row format
    joint_angle_metrics = {}

    # Step 4: Calculate max and release angles for each joint within the shooting motion
    for joint_name in joint_combinations.keys():
        max_angle = shooting_motion_df[f'{joint_name}_angle'].max()
        release_angle = (
            shooting_motion_df.at[release_frame_index, f'{joint_name}_angle']
            if release_frame_index in shooting_motion_df.index else np.nan
        )

        # Store these as single-row format columns
        joint_angle_metrics[f'{joint_name}_max_angle'] = max_angle
        joint_angle_metrics[f'{joint_name}_release_angle'] = release_angle

        if debug:
            print(f"Debug: {side} {joint_name} max_angle: {max_angle}, release_angle: {release_angle}")

    # Step 5: Convert metrics dictionary to a single-row DataFrame for output
    joint_angle_metrics_df = pd.DataFrame([joint_angle_metrics])

    # Final debug to check both DataFrames
    if debug:
        print("Debug: Main DataFrame with ongoing joint angles:\n", df.head())
        print("Debug: Joint Angle Metrics DataFrame (single row with max and release angles during shooting motion):\n", joint_angle_metrics_df)

    return df, joint_angle_metrics_df


if __name__ == "__main__":
    # Load and process data
    print("Debug: Loading and parsing file: ../../SPL-Open-Data/basketball/freethrow/data/P0001/BB_FT_P0001_T0001.json")
    data, trial_id, result, landing_x, landing_y, entry_angle, _ = load_single_ft_and_parse(
        "../../SPL-Open-Data/basketball/freethrow/data/P0001/BB_FT_P0001_T0001.json"
    )
    debug = True
    df = main_create_dataframe(data, trial_id, result, landing_x, landing_y, entry_angle, debug=False)
    
    
    df = calculate_ball_speed_velocity_direction(df, debug=False)
    df = main_label_shot_phases(df)
    release_frame_index = df.index[df['release_point_filter'] == 1].tolist()[0]
    
    df, release_index, projection_df, ml_metrics_df = main_ball_trajectory_analysis(df, 
                                                                                    release_frame_index,
                                                                                    debug=False)
    print("Final DataFrame with release features and trajectory data:\n", df)
    print("columns = ", df.columns)
    
    joints = [
        'L_ANKLE', 'R_ANKLE', 'L_KNEE', 'R_KNEE', 'L_HIP', 'R_HIP',
        'L_ELBOW', 'R_ELBOW', 'L_WRIST', 'R_WRIST',
        'L_1STFINGER', 'L_5THFINGER', 'R_1STFINGER', 'R_5THFINGER'
    ]

    # Step 1: Calculate joint power and metrics
    df, joint_power_metrics_df = main_calculate_joint_power(df, release_frame_index, debug=False)
    
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
