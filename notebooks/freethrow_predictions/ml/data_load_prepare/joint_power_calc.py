import numpy as np
import pandas as pd

def calculate_joint_speed(df, joint):
    """Calculate the speed for a specified joint over the entire DataFrame."""
    speed = np.sqrt(
        (df[f'{joint}_x'].diff() ** 2) +
        (df[f'{joint}_y'].diff() ** 2) +
        (df[f'{joint}_z'].diff() ** 2)
    ) / df['dt']
    return speed

def add_ongoing_power_columns(df, joints):
    """
    Adds ongoing power (speed) columns for each joint in the main DataFrame.
    """
    for joint in joints:
        if {f'{joint}_x', f'{joint}_y', f'{joint}_z'}.issubset(df.columns):
            df[f'{joint}_ongoing_power'] = calculate_joint_speed(df, joint)
        else:
            print(f"Warning: Missing coordinates for joint '{joint}'.")
    return df

def calculate_joint_power_metrics(df, joints, debug=False):
    """
    Calculate overall metrics for each joint's ongoing power during shooting motion.
    """
    joint_power_metrics = {}
    # Filter the DataFrame to include only shooting motion frames
    shooting_motion_df = df[df['shooting_motion'] == 1]
    
    for joint in joints:
        power_column = f'{joint}_ongoing_power'
        if power_column in df.columns:
            joint_power = shooting_motion_df[power_column]
            # Calculate metrics only during shooting motion
            joint_power_metrics.update({
                f'{joint}_min_power': joint_power.min(),
                f'{joint}_max_power': joint_power.max(),
                f'{joint}_avg_power': joint_power.mean(),
                f'{joint}_std_power': joint_power.std(),
            })
            if debug:
                print(f"Debug: Calculated metrics for {joint}: {joint_power_metrics}")
        else:
            print(f"Warning: {power_column} not found in DataFrame.")
    return pd.DataFrame([joint_power_metrics])

def main_calculate_joint_power(df, release_frame_index, debug=False):
    """
    Main function to calculate ongoing joint power and overall metrics.
    """
    joints = [
        'L_ANKLE', 'R_ANKLE', 'L_KNEE', 'R_KNEE', 'L_HIP', 'R_HIP',
        'L_ELBOW', 'R_ELBOW', 'L_WRIST', 'R_WRIST',
        'L_1STFINGER', 'L_5THFINGER', 'R_1STFINGER', 'R_5THFINGER'
    ]
    if debug:
        print("Debug: Starting joint power calculations...")
    
    # Step 1: Add ongoing power columns directly to df
    df = add_ongoing_power_columns(df, joints)
    
    # Step 2: Calculate joint power metrics during shooting motion
    joint_power_metrics_df = calculate_joint_power_metrics(df, joints, debug=debug)
    
    if debug:
        print("Debug: Joint power calculations completed.")
    
    # Return df with ongoing power columns and the metrics DataFrame
    return df, joint_power_metrics_df

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
    
    df, release_index, projection_df, ml_metrics_df = main_ball_trajectory_analysis(df, release_frame_index, debug=True)
    print("Final DataFrame with release features and trajectory data:\n", df)
    print("columns = ", df.columns)
    
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

