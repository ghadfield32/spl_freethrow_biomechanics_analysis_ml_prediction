import numpy as np

def calculate_ball_speed_velocity_direction(df, debug=False):
    # Calculate ball speed using 3D distance and time differences
    df['ball_speed'] = (np.sqrt(
        (df['ball_x'].diff()**2) +
        (df['ball_y'].diff()**2) +
        (df['ball_z'].diff()**2)
    ) + np.sqrt(
        (df['ball_x'].shift(-1).diff()**2) +
        (df['ball_y'].shift(-1).diff()**2) +
        (df['ball_z'].shift(-1).diff()**2)
    )) / (df['frame_time'].diff() + df['frame_time'].shift(-1).diff())

    # Calculate ball velocity components along x, y, z axes
    df['ball_velocity_x'] = (df['ball_x'].diff() / df['frame_time'].diff() +
                       df['ball_x'].shift(-1).diff() / df['frame_time'].shift(-1).diff()) / 2
    df['ball_velocity_y'] = (df['ball_y'].diff() / df['frame_time'].diff() +
                       df['ball_y'].shift(-1).diff() / df['frame_time'].shift(-1).diff()) / 2
    df['ball_velocity_z'] = (df['ball_z'].diff() / df['frame_time'].diff() +
                       df['ball_z'].shift(-1).diff() / df['frame_time'].shift(-1).diff()) / 2
    df['overall_ball_velocity'] = np.sqrt(df['ball_velocity_x']**2 + df['ball_velocity_y']**2 + df['ball_velocity_z']**2)

    # Calculate normalized direction components (unit vectors) along x, y, z
    df['ball_direction_x'] = df['ball_velocity_x'] / df['overall_ball_velocity']
    df['ball_direction_y'] = df['ball_velocity_y'] / df['overall_ball_velocity']
    df['ball_direction_z'] = df['ball_velocity_z'] / df['overall_ball_velocity']

    # Filter out rows with NaN direction values caused by zero velocity (overall_ball_velocity = 0)
    df = df.dropna(subset=['ball_direction_x', 'ball_direction_y', 'ball_direction_z']).reset_index(drop=True)

    # Debugging to verify direction accuracy
    if debug:
        print("Debug: Calculated ball speed, velocity, and direction.")
        print("Debug: NaN counts for key columns after filtering:")
        print(df[['ball_speed', 'ball_velocity_x', 'ball_velocity_y', 'ball_velocity_z', 'overall_ball_velocity', 'ball_direction_x', 'ball_direction_y', 'ball_direction_z']].isna().sum())
        
        # Verify direction correctness by comparing to velocity components
        df['computed_ball_velocity_x'] = df['ball_direction_x'] * df['overall_ball_velocity']
        df['computed_ball_velocity_y'] = df['ball_direction_y'] * df['overall_ball_velocity']
        df['computed_ball_velocity_z'] = df['ball_direction_z'] * df['overall_ball_velocity']
        
        # Check for discrepancies between original and computed velocity components
        discrepancy_x = np.abs(df['ball_velocity_x'] - df['computed_ball_velocity_x']).mean()
        discrepancy_y = np.abs(df['ball_velocity_y'] - df['computed_ball_velocity_y']).mean()
        discrepancy_z = np.abs(df['ball_velocity_z'] - df['computed_ball_velocity_z']).mean()
        
        print(f"Debug: Average discrepancy for ball_velocity_x: {discrepancy_x:.5f}")
        print(f"Debug: Average discrepancy for ball_velocity_y: {discrepancy_y:.5f}")
        print(f"Debug: Average discrepancy for ball_velocity_z: {discrepancy_z:.5f}")
        
        # Output sample data for manual inspection
        print("Sample data around release frame:")
        print(df[['frame_time', 'ball_speed', 'ball_velocity_x', 'computed_ball_velocity_x', 'ball_velocity_y', 'computed_ball_velocity_y', 'ball_velocity_z', 'computed_ball_velocity_z', 'overall_ball_velocity',
                  'ball_direction_x', 'ball_direction_y', 'ball_direction_z']].head(10))

    return df



# Test the function
if __name__ == "__main__":
    # from data_preprocessing.dataframe_creation import main_create_dataframe
    # from data_loading.load_and_parse import load_single_ft_and_parse
    data, trial_id, result, landing_x, landing_y, entry_angle, _ = load_single_ft_and_parse("../../SPL-Open-Data/basketball/freethrow/data/P0001/BB_FT_P0001_T0001.json")
    df = main_create_dataframe(data, trial_id, result, landing_x, landing_y, entry_angle, debug=False)
    df = calculate_ball_speed_velocity_direction(df, debug=False)

