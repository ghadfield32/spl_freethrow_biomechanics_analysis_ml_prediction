
import numpy as np
import pandas as pd
import math

def calculate_dt(df, time_column='frame_time', debug=False):
    """
    Calculate the time delta ('dt') between frames and handle initial NaN values.
    
    Parameters:
    - df: DataFrame containing the time column.
    - time_column: The column name containing time values in milliseconds.
    - debug: If True, prints detailed debug information and validation checks.
    
    Returns:
    - df with an added 'dt' column, forward-filled and validated.
    """
    # Calculate 'dt' as the difference in time between consecutive frames in seconds
    df['dt'] = df[time_column].diff() / 1000.0  # Convert ms to seconds
    
    # Check for initial NaN or zero values in 'dt' and forward fill them
    if pd.isna(df['dt'].iloc[1]) or df['dt'].iloc[1] <= 0:
        df['dt'] = df['dt'].ffill()

    # Handle the first row NaN specifically by assigning a default value if it persists
    if pd.isna(df.loc[0, 'dt']):
        df.loc[0, 'dt'] = df['dt'].iloc[1:].mean()  # Use mean of subsequent values as an estimate

    # Additional validation checks
    if debug:
        # Ensure there are no null values in 'dt' after filling
        if df['dt'].isnull().any():
            print("Warning: 'dt' still contains NaN values after forward fill.")
        else:
            print("Debug: No NaN values in 'dt' after forward fill.")

        # Ensure 'dt' values are all positive
        if (df['dt'] <= 0).any():
            print("Warning: 'dt' contains non-positive values. Check the time column for consistency.")
        else:
            print("Debug: All 'dt' values are positive.")

        # Output some sample values for verification
        print("Debug: 'dt' calculated with forward fill. Sample values:", df['dt'].head())

        # Check if 'dt' values are reasonably consistent (not fluctuating unexpectedly)
        dt_std = df['dt'].std()
        dt_mean = df['dt'].mean()
        if dt_std > dt_mean * 0.1:
            print(f"Warning: 'dt' values have high variance (std={dt_std:.4f}). This could indicate inconsistencies in time intervals.")

    return df




def calculate_ball_dynamics(df, release_frame_index, debug=False):
    """
    Calculate ball speed, velocity, and direction starting from the release frame.
    """
    # Set pre-release frames to NaN for dynamics
    df.loc[:release_frame_index, ['ball_velocity_x', 'ball_velocity_y', 'ball_velocity_z', 'ball_speed', 'ball_direction_x', 'ball_direction_y', 'ball_direction_z']] = np.nan

    # Calculate 'dt' using the helper function
    df = calculate_dt(df, time_column='frame_time', debug=debug)

    # Calculate differences for dynamics
    df['dx'] = df['ball_x'].diff()
    df['dy'] = df['ball_y'].diff()
    df['dz'] = df['ball_z'].diff()

    # Calculate velocities and speed
    df['ball_velocity_x'] = df['dx'] / df['dt']
    df['ball_velocity_y'] = df['dy'] / df['dt']
    df['ball_velocity_z'] = df['dz'] / df['dt']
    df['ball_speed'] = np.sqrt(df['ball_velocity_x']**2 + df['ball_velocity_y']**2 + df['ball_velocity_z']**2)
    df['ball_direction_x'] = df['ball_velocity_x'] / df['ball_speed']
    df['ball_direction_y'] = df['ball_velocity_y'] / df['ball_speed']
    df['ball_direction_z'] = df['ball_velocity_z'] / df['ball_speed']
    df[['ball_direction_x', 'ball_direction_y', 'ball_direction_z']] = df[['ball_direction_x', 'ball_direction_y', 'ball_direction_z']].fillna(0)

    return df



def extract_release_features(df, release_frame_index, debug=False):
    """
    Extracts key features at the release frame for machine learning and visualization.
    """
    if release_frame_index is None:
        if debug:
            print("Debug: No release frame found.")
        return pd.DataFrame()  # Return empty DataFrame if no release frame

    release_row = df.iloc[release_frame_index]

    release_features = {
        'release_ball_speed': release_row['ball_speed'],
        'release_ball_velocity_x': release_row['ball_velocity_x'],
        'release_ball_velocity_y': release_row['ball_velocity_y'],
        'release_ball_velocity_z': release_row['ball_velocity_z'],
        'release_ball_direction_x': release_row['ball_direction_x'],
        'release_ball_direction_y': release_row['ball_direction_y'],
        'release_ball_direction_z': release_row['ball_direction_z'],
        'release_ball_x': release_row['ball_x'],
        'release_ball_y': release_row['ball_y'],
        'release_ball_z': release_row['ball_z'],
        'release_frame_time': release_row['frame_time']
    }

    # Calculate release angle
    horizontal_velocity = np.hypot(release_features['release_ball_velocity_x'], release_features['release_ball_velocity_y'])
    release_features['release_angle'] = math.degrees(math.atan2(release_features['release_ball_velocity_z'], horizontal_velocity))

    # Calculate time to peak height
    g = -9.81  # Gravity in METERS
    release_features['time_to_peak'] = -release_features['release_ball_velocity_z'] / g

    # Calculate peak height relative to release height
    peak_height = release_features['release_ball_z'] + (release_features['release_ball_velocity_z'] ** 2) / (2 * -g)
    release_features['peak_height_relative'] = peak_height - release_features['release_ball_z']

    return pd.DataFrame([release_features])  # Return as a single-row DataFrame



def project_ball_trajectory(df, release_index, debug=False):
    """
    Project the ball's trajectory based on release dynamics.
    """
    g = -9.81  # Gravity in METERS
    release_row = df.iloc[release_index]

    vx = release_row['ball_velocity_x']
    vy = release_row['ball_velocity_y']
    vz = release_row['ball_velocity_z']
    x0, y0, z0 = release_row['ball_x'], release_row['ball_y'], release_row['ball_z']

    projection_time = np.linspace(0, 2, num=200)
    dt = projection_time[1] - projection_time[0]

    x_proj = x0 + vx * projection_time
    y_proj = y0 + vy * projection_time
    z_proj = z0 + vz * projection_time + 0.5 * g * projection_time**2

    valid_indices = z_proj >= 0  # Keep trajectory points above ground level
    x_proj = x_proj[valid_indices]
    y_proj = y_proj[valid_indices]
    z_proj = z_proj[valid_indices]
    proj_time = projection_time[valid_indices]

    if debug:
        print("Debug: Projected ball trajectory up to impact point.")

    return proj_time, x_proj, y_proj, z_proj


def main_ball_trajectory_analysis(df, release_frame_index, debug=False):
    """
    Main analysis function to calculate dynamics and extract release features.
    """
    if release_frame_index is None:
        if debug:
            print("Debug: No release frame found.")
        return df, None, None, None  # Return placeholders for consistency

    df = calculate_ball_dynamics(df, release_frame_index, debug=debug)

    if debug:
        print("Debug: Release frame index:", release_frame_index)
        print("Debug: Frame data at release:\n", df.loc[release_frame_index])

    # Extract release features for ML
    ml_metrics_df = extract_release_features(df, release_frame_index, debug=debug)
    proj_time, x_proj, y_proj, z_proj = project_ball_trajectory(df, release_frame_index, debug=debug)

    # Projection results for visualization
    projection_df = pd.DataFrame({
        'projection_time': proj_time,
        'projected_x': x_proj,
        'projected_y': y_proj,
        'projected_z': z_proj
    })

    return df, release_frame_index, projection_df, ml_metrics_df


# Example usage
if __name__ == "__main__":
    # Placeholder functions for data loading and creation
    data, trial_id, result, landing_x, landing_y, entry_angle, _ = load_single_ft_and_parse(
        "../../SPL-Open-Data/basketball/freethrow/data/P0001/BB_FT_P0001_T0001.json"
    )
    df = main_create_dataframe(data, trial_id, result, landing_x, landing_y, entry_angle, debug=False)
    df = calculate_ball_speed_velocity_direction(df, debug=False)
    df = main_label_shot_phases(df, debug=False)
    release_frame_index = df.index[df['release_point_filter'] == 1].tolist()[0]

    df, release_index, projection_df, ml_metrics_df = main_ball_trajectory_analysis(df, release_frame_index, debug=True)
    print("Final DataFrame with release features and trajectory data:\n", df)
    print("ML Metrics DataFrame for model input:\n", ml_metrics_df)
