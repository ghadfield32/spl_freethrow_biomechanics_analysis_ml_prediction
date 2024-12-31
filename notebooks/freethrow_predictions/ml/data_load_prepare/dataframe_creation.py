
import pandas as pd
import numpy as np

def create_dataframe(data, trial_id, result, landing_x, landing_y, entry_angle, debug=False):
    frame_data = []
    if debug:
        print("Debug: Initializing dataframe creation with given trial metadata.")

    # Process each frame in the data
    for i, frame in enumerate(data['tracking']):
        frame_time = frame['time']
        ball_pos = frame['data'].get('ball', [None, None, None])
        player_pos = frame['data']['player']

        if debug:
            print(f"Debug: Processing frame {i} with time {frame_time}. Ball position: {ball_pos}")

        # Flattening frame data with player and ball positions
        flat_frame = {
            'trial_id': trial_id,
            'result': result,
            'landing_x': landing_x,
            'landing_y': landing_y,
            'entry_angle': entry_angle,
            'frame_time': frame_time,
            'ball_x': ball_pos[0] if ball_pos[0] is not None else np.nan,
            'ball_y': ball_pos[1] if ball_pos[1] is not None else np.nan,
            'ball_z': ball_pos[2] if ball_pos[2] is not None else np.nan,
        }

        for part, coords in player_pos.items():
            flat_frame[f'{part}_x'] = coords[0]
            flat_frame[f'{part}_y'] = coords[1]
            flat_frame[f'{part}_z'] = coords[2]

        frame_data.append(flat_frame)
    
    df = pd.DataFrame(frame_data)
    if debug:
        print("Debug: DataFrame created from frame data. Dimensions:", df.shape)

    # Set the index to the sequential frame number
    df.index = np.arange(len(df))
    if debug:
        print(f"Debug: DataFrame index set to sequential frame numbers. Index range: {df.index.min()} to {df.index.max()}")

    # Filter out rows where the ball is not being tracked
    original_len = len(df)
    df.dropna(subset=['ball_x', 'ball_y', 'ball_z'], inplace=True)
    filtered_len = len(df)
    df.reset_index(drop=True, inplace=True)
    if debug:
        print(f"Debug: DataFrame filtered to exclude rows where ball is not tracked. Rows before: {original_len}, after: {filtered_len}")

    if debug:
        # Contextual debug output
        print(f"Debug: Created DataFrame for Trial ID: {trial_id}")
        print("Debug: Columns available in DataFrame:", df.columns)
        print("\nDebug: Data types of each column for validation:", df.dtypes)
        print("\nDebug: Sample row from DataFrame to inspect initial data structure:", df.iloc[0] if not df.empty else "DataFrame is empty.")
        print("\nDebug: Count of null values in each column to check data completeness:", df.isna().sum())
        print("\nDebug: Summary statistics for numeric columns, providing insight into data range and variance:", df.describe())

    return df


def main_create_dataframe(data, trial_id, result, landing_x, landing_y, entry_angle, debug=False):
    if debug:
        print("Debug: Calling main_create_dataframe with trial data and parameters.")
    df = create_dataframe(data, trial_id, result, landing_x, landing_y, entry_angle, debug=debug)
    return df


# Test the function
if __name__ == "__main__":
    # from data_loading.load_and_parse import load_single_ft_and_parse
    data, trial_id, result, landing_x, landing_y, entry_angle, _ = load_single_ft_and_parse("../../SPL-Open-Data/basketball/freethrow/data/P0001/BB_FT_P0001_T0001.json", debug=False)
    main_create_dataframe(data, trial_id, result, landing_x, landing_y, entry_angle, debug=False)
