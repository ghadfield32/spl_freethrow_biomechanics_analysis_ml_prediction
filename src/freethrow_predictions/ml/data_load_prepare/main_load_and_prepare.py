
import os
import pandas as pd
import json
from data_load_prepare.load_and_parse import load_single_ft_and_parse
from data_load_prepare.dataframe_creation import main_create_dataframe
from data_load_prepare.velocity_and_speed_calc import calculate_ball_speed_velocity_direction
from data_load_prepare.phase_labeling import main_label_shot_phases
from data_load_prepare.ball_trajectory_and_release_time_stats import main_ball_trajectory_analysis
from data_load_prepare.joint_power_calc import main_calculate_joint_power
from data_load_prepare.joint_angles_details import  calculate_joint_angles_over_motion
from data_load_prepare.key_feature_extraction import main_prepare_ml_data, load_player_info, get_column_definitions

def process_file(file_path, shot_id, player_info, debug=False):
    # Load and parse JSON data
    data, trial_id, result, landing_x, landing_y, entry_angle, _ = load_single_ft_and_parse(file_path)
    if data is None:
        if debug:
            print(f"Debug: Skipping file {file_path} due to parsing error.")
        return pd.DataFrame(), pd.DataFrame()

    # Step 1: Create initial DataFrame from parsed data
    df = main_create_dataframe(data, trial_id, result, landing_x, landing_y, entry_angle, debug=debug)
    df = calculate_ball_speed_velocity_direction(df, debug=debug)
    df = main_label_shot_phases(df)

    # Step 2: Identify release frame
    release_frame_index = df.index[df['release_point_filter'] == 1].tolist()[0] if df['release_point_filter'].sum() > 0 else None
    if release_frame_index is None:
        if debug:
            print(f"Debug: No release frame found for file {file_path}.")
        return pd.DataFrame(), pd.DataFrame()

    # Step 3: Calculate ball dynamics and release metrics first (for `dt` and velocity/direction data)
    df, _, _, ml_metrics_df = main_ball_trajectory_analysis(df, release_frame_index, debug=debug)

    # Step 4: Calculate joint power and joint angle metrics
    df, joint_power_metrics_df = main_calculate_joint_power(df, release_frame_index, debug=debug)
    df, joint_angle_metrics_df = calculate_joint_angles_over_motion(df, release_frame_index, debug=debug)

    # Step 5: Prepare ML features combining shot details, joint power, angles, release metrics, and player info
    key_features_dataframe = main_prepare_ml_data(df, joint_power_metrics_df, joint_angle_metrics_df, ml_metrics_df, player_info, debug=debug)
    key_features_dataframe['trial_id'] = trial_id
    key_features_dataframe['shot_id'] = shot_id

    return df, key_features_dataframe


def bulk_process_directory(directory_path, player_info_path, debug=False):
    # Load player information
    player_info = load_player_info(player_info_path)
    if debug:
        print("Debug: Loaded player info:\n", player_info)

    all_granular_data = []
    all_features = []
    shot_id = 1
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            if debug:
                print(f"Debug: Processing file: {file_path}")

            granular_df, trial_features_df = process_file(file_path, shot_id, player_info, debug=debug)
            shot_id += 1
            
            if not granular_df.empty and not trial_features_df.empty:
                all_granular_data.append(granular_df)
                all_features.append(trial_features_df)

    final_granular_df = pd.concat(all_granular_data, ignore_index=True) if all_granular_data else pd.DataFrame()
    final_ml_df = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
    
    if debug:
        print("Debug: Final granular DataFrame created with shape:", final_granular_df.shape)
        print("Debug: Final ML DataFrame created with shape:", final_ml_df.shape)
    
    return final_granular_df, final_ml_df

if __name__ == "__main__":
    directory_path = "../../../../SPL-Open-Data/basketball/freethrow/data/P0001"
    player_info_path = "../../../../SPL-Open-Data/basketball/freethrow/participant_information.json"

    # Run the bulk processing with player info integration
    final_granular_df, final_ml_df = bulk_process_directory(directory_path, player_info_path, debug=False)
    
    if not final_granular_df.empty and not final_ml_df.empty:
        print("Debug: Granular free throw data for all trials:")
        print(final_granular_df.head())
        print("Debug: ML Features DataFrame for all trials with player info:")
        print(final_ml_df.head())

        # Column Definitions
        column_definitions = get_column_definitions()
        print("Column Definitions for the Dataset:\n", column_definitions)
