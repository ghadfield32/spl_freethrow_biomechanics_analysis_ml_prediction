
import logging
import pandas as pd
from IPython.display import display

from animate.animate_from_df import animate_trial_from_df

joint_configs = [
    {
        'name': 'knee',
        'min_angle_key': 'max_knee_angle_filtered_optimal_min',
        'max_angle_key': 'max_knee_angle_filtered_optimal_max',
        'release_min_angle_key': 'release_knee_angle_filtered_optimal_min',
        'release_max_angle_key': 'release_knee_angle_filtered_optimal_max',
        'angle_key': 'knee_angle',
        'is_max_key': 'is_max_knee_angle',
        'classification_key': 'max_knee_shot_classification',
        'release_classification_key': 'release_knee_shot_classification'
    },
    {
        'name': 'elbow',
        'min_angle_key': 'max_elbow_angle_filtered_optimal_min',
        'max_angle_key': 'max_elbow_angle_filtered_optimal_max',
        'release_min_angle_key': 'release_elbow_angle_filtered_optimal_min',
        'release_max_angle_key': 'release_elbow_angle_filtered_optimal_max',
        'angle_key': 'elbow_angle',
        'is_max_key': 'is_max_elbow_angle',
        'classification_key': 'max_elbow_shot_classification',
        'release_classification_key': 'release_elbow_shot_classification'
    },
    {
        'name': 'wrist',
        'min_angle_key': 'max_wrist_angle_filtered_optimal_min',
        'max_angle_key': 'max_wrist_angle_filtered_optimal_max',
        'release_min_angle_key': 'release_wrist_angle_filtered_optimal_min',
        'release_max_angle_key': 'release_wrist_angle_filtered_optimal_max',
        'angle_key': 'wrist_angle',
        'is_max_key': 'is_max_wrist_angle',
        'classification_key': 'max_wrist_shot_classification',
        'release_classification_key': 'release_wrist_shot_classification'
    }
]


# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Load data from the specified CSV file
        data_path = "../../data/processed/final_granular_dataset.csv"
        final_granular_df_with_stats = pd.read_csv(data_path)
        logger.debug(f"Loaded data from {data_path} with shape {final_granular_df_with_stats.shape}")

        # Define connections between joints
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
        logger.debug("Defined joint connections for player skeleton.")

        # Select a specific trial for visualization
        trial_id_to_visualize = 'T0088'  # Replace with actual trial ID you want to visualize
        trial_data = final_granular_df_with_stats[final_granular_df_with_stats['trial_id'] == trial_id_to_visualize]
        trial_data = trial_data.sort_values(by='frame_time').reset_index(drop=True)
        logger.debug(f"Selected trial ID '{trial_id_to_visualize}' with {len(trial_data)} frames.")

        # Determine the release frame (the frame where release_point_filter is 1)
        release_frames = trial_data.index[trial_data["release_point_filter"] == 1].tolist()
        release_frame = release_frames[0] if release_frames else None
        if release_frame is not None:
            logger.debug(f"Release frame found at index {release_frame}.")
        else:
            logger.warning("No release frame found in the trial data.")

        # Set parameters for visualization
        viewpoint_name = "diagonal_player_centric"  # Choose from COMMON_VIEWPOINTS
        zlim = 15        # Adjust for height


        # Call the first animation function
        animation_html = animate_trial_from_df(
            df=trial_data,
            release_frame=release_frame,
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
        
        # Display the first animation
        display(animation_html)

    except Exception as e:
        logger.error(f"An error occurred in main: {e}")
        raise

if __name__ == "__main__":
    main()
