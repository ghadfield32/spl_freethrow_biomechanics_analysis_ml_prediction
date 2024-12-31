
"""
Module: feature_engineering_optimal_release_angle_metrics

Goal: Optimal release angle shooting feature engineering for basketball free throw predictions.

This module processes trial data to compute and validate optimal release angles based on theoretical models.
It includes functions to create reference tables, compute release angles, validate against models, and merge results
into granular and ML datasets.
"""

import numpy as np
import pandas as pd
import logging

# Configure logging
def configure_logging(debug: bool):
    """
    Configure logging based on the debug flag.

    Parameters:
    - debug: Boolean flag to set logging level.

    Returns:
    - logger: Configured logger instance.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

# Initialize logger with default debug=False
logger = configure_logging(debug=False)

def create_optimal_angle_reference_table(debug: bool = False):
    """
    Create the expanded reference table for optimal release angles based on release height and shot distance.

    Parameters:
    - debug: Whether to enable debug output.

    Returns:
    - reference_df: DataFrame where rows represent release heights and columns represent distances.
    """
    global logger
    try:
        data = {
            9: [62.92, 61.85, 60.71, 59.53, 58.28, 56.98, 55.63, 54.22, 52.76, 51.26, 49.73, 48.17, 46.59, 45.00],
            11: [60.29, 59.31, 58.28, 57.22, 56.12, 54.99, 53.83, 52.63, 51.40, 50.15, 48.88, 47.60, 46.30, 45.00],
            14: [57.45, 56.60, 55.72, 54.83, 53.91, 52.97, 52.02, 51.05, 50.06, 49.07, 48.06, 47.04, 46.02, 45.00],
            17: [55.46, 54.72, 53.96, 53.19, 52.41, 51.62, 50.82, 50.00, 49.18, 48.35, 47.52, 46.68, 45.84, 45.00],
            20: [54.00, 53.35, 52.69, 52.02, 51.34, 50.65, 49.96, 49.27, 48.56, 47.86, 47.14, 46.43, 45.72, 45.00],
            24: [52.58, 52.02, 51.45, 50.88, 50.31, 49.73, 49.15, 48.56, 47.97, 47.38, 46.79, 46.19, 45.60, 45.00]
        }
        heights = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]  # Heights in feet

        reference_df = pd.DataFrame(data, index=heights)
        reference_df.index.name = "Release Height (ft)"
        reference_df.columns.name = "Distance to Basket (ft)"

        if debug:
            logger.debug("Reference table created successfully.")
            logger.debug(f"Reference Table Shape: {reference_df.shape}")
            logger.debug("New Columns Added:")
            logger.debug(f"Columns: {reference_df.columns.tolist()} | Data Types: {reference_df.dtypes.to_dict()}")

        else:
            logger.info("Reference table created successfully.")

        return reference_df

    except Exception as e:
        logger.error(f"Error in create_optimal_angle_reference_table: {e}")
        return pd.DataFrame()

def get_optimal_angle(reference_df: pd.DataFrame, release_height: float, distance_to_basket: float = 15, debug: bool = False):
    """
    Get the optimal release angle using the reference table with interpolation.

    Parameters:
    - reference_df: DataFrame with optimal release angles.
    - release_height: Release height in feet.
    - distance_to_basket: Distance to basket in feet.
    - debug: Whether to enable debug output.

    Returns:
    - optimal_angle: The interpolated optimal release angle.
    """
    global logger
    try:
        interpolated_angles = reference_df.apply(
            lambda col: np.interp(release_height, reference_df.index, col)
        )
        optimal_angle = np.interp(distance_to_basket, interpolated_angles.index, interpolated_angles.values)

        if debug:
            logger.debug(f"Interpolated Angles Shape: {interpolated_angles.shape}")
            logger.debug("Interpolated Angles Summary:")
            logger.debug(f"Data Types: {interpolated_angles.dtypes}")
            logger.debug(f"Sample Values: {interpolated_angles.head().to_dict()}")

            logger.debug(f"Optimal Release Angle for Distance {distance_to_basket} ft: {optimal_angle:.2f}°")


        return optimal_angle

    except Exception as e:
        logger.error(f"Error in get_optimal_angle: {e}")
        return None

def compute_averaged_velocities(trial_data: pd.DataFrame, release_frame: int, frames_to_average: int = 3, debug: bool = False):
    """
    Compute the averaged velocities over multiple frames post-release.

    Parameters:
    - trial_data: DataFrame containing trial data.
    - release_frame: Index of the release frame.
    - frames_to_average: Number of frames to average post-release.
    - debug: Whether to enable debug output.

    Returns:
    - avg_velocity_x: Averaged velocity in the x-direction.
    - avg_velocity_z: Averaged velocity in the z-direction.
    """
    global logger
    try:
        post_release_frames = trial_data.loc[release_frame:release_frame + frames_to_average]

        avg_velocity_x = post_release_frames['ball_velocity_x'].mean()
        avg_velocity_z = post_release_frames['ball_velocity_z'].mean()

        if debug:
            logger.debug(f"Averaged Velocities Shape: {post_release_frames.shape}")
            logger.debug("Averaged Velocities Summary:")
            logger.debug(f"avg_velocity_x: {avg_velocity_x:.2f} ft/s | avg_velocity_z: {avg_velocity_z:.2f} ft/s")


        return avg_velocity_x, avg_velocity_z

    except Exception as e:
        logger.error(f"Error in compute_averaged_velocities: {e}")
        return None, None

def validate_against_theoretical_models(release_angle: float, optimal_release_angle: float, shot_outcome: int, debug: bool = False):
    """
    Validate calculated release parameters against theoretical models and shot outcomes.

    Parameters:
    - release_angle: Calculated release angle.
    - optimal_release_angle: Optimal release angle from theoretical models.
    - shot_outcome: Whether the shot was made (1) or missed (0).
    - debug: Whether to enable debug output.

    Returns:
    - None
    """
    global logger
    try:
        if debug:
            logger.debug("Validation Against Theoretical Models:")
            logger.debug(f"Release Angle Shape: N/A")
            logger.debug("New Columns Added: N/A")
            logger.debug(f"Actual Release Angle: {release_angle:.2f}°")
            logger.debug(f"Optimal Release Angle: {optimal_release_angle:.2f}°")
            logger.debug(f"Shot Outcome: {'Made' if shot_outcome == 1 else 'Missed'}")

        discrepancy = abs(optimal_release_angle - release_angle)
        if discrepancy > 5:
            logger.warning(f"Significant discrepancy of {discrepancy:.2f}° between actual and optimal release angles.")
            logger.warning("Consider revising calculations or incorporating additional factors (e.g., air resistance).")
        else:
            if debug:
                logger.debug(f"Discrepancy ({discrepancy:.2f}°) within acceptable range.")

    except Exception as e:
        logger.error(f"Error in validate_against_theoretical_models: {e}")

def analyze_release_all_trials(data: pd.DataFrame, reference_df: pd.DataFrame, distance_to_basket: float = 15, debug: bool = False):
    """
    Analyze the release angle and compare it with the optimal angle for all trials.
    Add results (angles and metrics) as new columns for the entire DataFrame.

    Parameters:
    - data: Full DataFrame containing all trial data.
    - reference_df: DataFrame with optimal release angles.
    - distance_to_basket: Distance from the player to the basket (ft).
    - debug: Whether to enable debug output.

    Returns:
    - data: Updated DataFrame with added columns for release analysis.
    """
    global logger
    try:
        # Initialize new columns with NaN
        new_columns = {
            'initial_release_angle': np.nan,
            'calculated_release_angle': np.nan,
            'angle_difference': np.nan,
            'distance_to_basket': distance_to_basket,
            'optimal_release_angle': np.nan
        }
        for col in new_columns:
            data[col] = new_columns[col]

        if debug:
            logger.debug(f"Added new columns: {list(new_columns.keys())}")
            logger.debug(f"DataFrame Shape after adding columns: {data.shape}")
        else:
            logger.info("Initialized new columns for release analysis.")

        # Iterate over each unique trial_id
        unique_trial_ids = data['trial_id'].unique()
        if debug:
            logger.debug(f"Number of unique trial IDs to process: {len(unique_trial_ids)}")

        # Initialize a counter
        processed_trials = 0

        for trial_id in unique_trial_ids:
            trial_data = data[data['trial_id'] == trial_id]

            # Increment the counter
            processed_trials += 1

            # Log detailed debug information for the first trial only
            log_detailed_debug = debug and (processed_trials == 1)

            if log_detailed_debug:
                logger.debug(f"=== Processing Trial ID: {trial_id} ===")

            # Identify the release frame for this trial
            release_frames = trial_data.index[trial_data['release_point_filter'] == 1].tolist()
            release_frame = release_frames[0] if release_frames else None
            if release_frame is None:
                logger.warning(f"No release frame identified for trial {trial_id}. Skipping.")
                continue

            if log_detailed_debug:
                logger.debug(f"Release Frame for Trial {trial_id}: {release_frame}")

            # Ensure required columns are present
            required_columns = ['ball_x', 'ball_y', 'ball_z', 'player_height_ft', 'ball_velocity_x', 'ball_velocity_z']
            missing_columns = [col for col in required_columns if col not in trial_data.columns]

            if missing_columns:
                logger.error(f"Missing Required Columns for trial {trial_id}: {missing_columns}")
                continue

            # Calculate release metrics
            release_height = trial_data.at[release_frame, 'ball_z']
            player_height = trial_data.at[release_frame, 'player_height_ft']

            if log_detailed_debug:
                logger.debug(f"Release Height: {release_height:.2f} ft")
                logger.debug(f"Player Height: {player_height:.2f} ft")

            # Calculate initial release angle
            ball_velocity_x = trial_data.at[release_frame, 'ball_velocity_x']
            ball_velocity_z = trial_data.at[release_frame, 'ball_velocity_z']
            initial_release_angle = np.degrees(np.arctan2(ball_velocity_z, ball_velocity_x))

            if log_detailed_debug:
                logger.debug(f"Initial Release Angle: {initial_release_angle:.2f}°")

            # Compute averaged velocities for calculated release angle
            avg_velocity_x, avg_velocity_z = compute_averaged_velocities(
                trial_data, release_frame, frames_to_average=3, debug=log_detailed_debug
            )
            if avg_velocity_x is None or avg_velocity_z is None:
                logger.error(f"Insufficient data to compute averaged velocities for trial {trial_id}.")
                continue

            calculated_release_angle = np.degrees(np.arctan2(avg_velocity_z, avg_velocity_x))

            if log_detailed_debug:
                logger.debug(f"Calculated Release Angle: {calculated_release_angle:.2f}°")

            # Calculate optimal release angle (only once per trial)
            optimal_release_angle = get_optimal_angle(
                reference_df, release_height, distance_to_basket, debug=log_detailed_debug
            )

            if optimal_release_angle is None:
                logger.error(f"Optimal release angle could not be determined for trial {trial_id}.")
                continue

            if log_detailed_debug:
                logger.debug(f"Optimal Release Angle: {optimal_release_angle:.2f}°")

            # Compute angle difference (direct subtraction)
            angle_difference = calculated_release_angle - optimal_release_angle

            if log_detailed_debug:
                logger.debug(f"Angle Difference: {angle_difference:.2f}°")

            # Populate new columns for all rows with the current trial_id
            data.loc[data['trial_id'] == trial_id, 'initial_release_angle'] = initial_release_angle
            data.loc[data['trial_id'] == trial_id, 'calculated_release_angle'] = calculated_release_angle
            data.loc[data['trial_id'] == trial_id, 'angle_difference'] = angle_difference
            data.loc[data['trial_id'] == trial_id, 'optimal_release_angle'] = optimal_release_angle

            if log_detailed_debug:
                logger.debug(f"Updated columns for Trial ID: {trial_id}")

            # Optional: Validate against theoretical models if shot outcome is available
            if 'shot_outcome' in trial_data.columns:
                shot_outcome = trial_data['shot_outcome'].iloc[0]  # Assuming same outcome per trial
                validate_against_theoretical_models(
                    calculated_release_angle,
                    optimal_release_angle,
                    shot_outcome,
                    debug=log_detailed_debug
                )

        if debug:
            logger.debug(f"Completed analysis for all trials. Final DataFrame Shape: {data.shape}")
        else:
            logger.info("Completed analysis for all trials.")

        return data  # Ensure this return is present outside the try-except block

    except Exception as e:
        logger.error(f"Error in analyze_release_all_trials: {e}")  # Corrected function name
        return data  # Ensure that even in exception, a DataFrame is returned


def check_duplicates(df: pd.DataFrame, df_name: str, debug: bool = False):
    """
    Check for duplicate trial_ids in the DataFrame.

    Parameters:
    - df: The DataFrame to check.
    - df_name: Name of the DataFrame (for logging purposes).
    - debug: Whether to enable debug output.

    Returns:
    - None
    """
    global logger
    try:
        duplicate_ids = df['trial_id'][df['trial_id'].duplicated()].unique()
        df_shape = df.shape

        if len(duplicate_ids) > 0:
            logger.warning(f"Duplicate trial_ids found in {df_name}: {duplicate_ids}")
            if debug:
                logger.debug(f"DataFrame Shape: {df_shape}")
        else:
            if debug:
                logger.debug(f"DataFrame Shape: {df_shape} | No duplicate trial_ids found in {df_name}.")
            else:
                logger.info(f"No duplicate trial_ids found in {df_name}.")

    except Exception as e:
        logger.error(f"Error in check_duplicates: {e}")

def log_trial_ids(data: pd.DataFrame, stage: str, debug: bool = False):
    """
    Log the unique trial IDs and their counts.

    Parameters:
    - data: DataFrame containing trial data.
    - stage: A string indicating the processing stage.
    - debug: Whether to enable debug output.

    Returns:
    - None
    """
    global logger
    try:
        trial_ids = data['trial_id'].unique()
        trial_id_counts = data['trial_id'].value_counts()
        if debug:
            logger.debug(f"=== Trial IDs at {stage} ===")
            logger.debug(f"Unique Trial IDs ({len(trial_ids)}): {trial_ids.tolist()}")
            logger.debug(f"Trial ID Counts: {trial_id_counts.to_dict()}")
        else:
            logger.info(f"Logged trial IDs at {stage}.")

    except Exception as e:
        logger.error(f"Error in log_trial_ids: {e}")


def aggregate_angles(granular_data: pd.DataFrame, debug: bool = False):
    """
    Aggregate the computed angles per trial.

    Parameters:
    - granular_data: DataFrame containing granular data with computed angles.
    - debug: Whether to enable debug output.

    Returns:
    - aggregated_df: DataFrame with one row per trial_id containing the angles and angle_difference.
    """
    global logger
    try:
        aggregated_df = granular_data.groupby('trial_id').agg({
            'initial_release_angle': 'first',
            'calculated_release_angle': 'first',
            'optimal_release_angle': 'first',
            'angle_difference': 'first'
        }).reset_index()

        if debug:
            logger.debug(f"Aggregated angles Shape: {aggregated_df.shape}")
            logger.debug("Aggregated Angles Columns and Data Types:")
            logger.debug(f"{aggregated_df.dtypes.to_dict()}")

        else:
            logger.info("Aggregated angles per trial successfully.")

        return aggregated_df

    except Exception as e:
        logger.error(f"Error in aggregate_angles: {e}")
        return pd.DataFrame()

def merge_final_ml_dataset(final_ml_df: pd.DataFrame, aggregated_angles_df: pd.DataFrame, debug: bool = False):
    """
    Merge the aggregated angles into the final ML dataset by trial_id.

    Parameters:
    - final_ml_df: DataFrame representing the final ML dataset.
    - aggregated_angles_df: DataFrame with aggregated angles per trial_id.
    - output_filename: The name of the output CSV file.
    - debug: Whether to enable debug output.

    Returns:
    - merged_df: The merged DataFrame.
    """
    global logger
    try:
        # Capture the shape before merging
        initial_shape = final_ml_df.shape

        # Merge on 'trial_id'
        merged_df = final_ml_df.merge(
            aggregated_angles_df,
            on='trial_id',
            how='left',
            validate='one_to_one'
        )

        # Capture the shape after merging
        final_shape = merged_df.shape

        if debug:
            logger.debug(f"Initial final_ml_df shape: {initial_shape}")
            logger.debug(f"Final merged_df shape: {final_shape}")

        # Check if any rows were added or removed
        if initial_shape[0] != final_shape[0]:
            logger.error(f"Row count mismatch after merging: Before={initial_shape[0]}, After={final_shape[0]}")
        else:
            if debug:
                logger.debug(f"Merge successful. Row count remains the same: {final_shape[0]} rows.")
            else:
                logger.info("Merge successful. Row count remains unchanged.")

        # Check for any NaN values in the merged angle columns
        angle_columns = ['initial_release_angle', 'calculated_release_angle', 'optimal_release_angle', 'angle_difference']
        missing_angles = merged_df[angle_columns].isna().any(axis=1)
        num_missing = missing_angles.sum()
        if num_missing > 0:
            logger.warning(f"{num_missing} trials in final_ml_df did not receive angle data.")
            if debug:
                logger.debug(f"Trials missing angle data: {merged_df.loc[missing_angles, 'trial_id'].unique()}")
        else:
            if debug:
                logger.debug("All trials in final_ml_df have corresponding angle data.")
            else:
                logger.info("All trials in final_ml_df have corresponding angle data.")

        return merged_df

    except Exception as e:
        logger.error(f"Error merging final_ml_df with aggregated angles: {e}")
        return final_ml_df

def compare_unique_trial_ids(original_df: pd.DataFrame, processed_df: pd.DataFrame, stage: str, debug: bool = False):
    """
    Compare unique trial IDs between two DataFrames.

    Parameters:
    - original_df: Original DataFrame before processing.
    - processed_df: Processed DataFrame after certain operations.
    - stage: A string indicating the comparison stage.
    - debug: Whether to enable debug output.

    Returns:
    - None
    """
    global logger
    try:
        original_ids = set(original_df['trial_id'])
        processed_ids = set(processed_df['trial_id'])
        missing_ids = original_ids - processed_ids
        extra_ids = processed_ids - original_ids

        if missing_ids:
            logger.error(f"Missing Trial IDs after {stage}: {missing_ids}")
        if extra_ids:
            logger.error(f"Extra Trial IDs after {stage}: {extra_ids}")
        if not missing_ids and not extra_ids:
            if debug:
                logger.debug(f"All Trial IDs are consistent after {stage}.")
            else:
                logger.info(f"All Trial IDs are consistent after {stage}.")

    except Exception as e:
        logger.error(f"Error in compare_unique_trial_ids: {e}")

def add_optimized_angles_to_granular(final_granular_df: pd.DataFrame, final_ml_df: pd.DataFrame, reference_df: pd.DataFrame, debug: bool = False):
    """
    Add optimized angles to the granular dataset.

    Parameters:
    - final_granular_df: DataFrame containing granular trial data.
    - final_ml_df: DataFrame containing ML data with player heights.
    - reference_df: DataFrame containing the reference table for optimal angles.
    - debug: Whether to display the outputs for debugging.

    Returns:
    - final_granular_df_with_optimal_release_angle: Updated granular dataset with optimized angles.
    """
    global logger
    try:
        # Merge player height into granular dataset
        merged_df = final_granular_df.merge(
            final_ml_df[['trial_id', 'player_height_in_meters']],
            on='trial_id',
            how='left',
            validate='many_to_one'
        )
        merged_df['player_height_ft'] = merged_df['player_height_in_meters'] * 3.28084

        if debug:
            logger.debug("Merged player_height_in_meters into granular dataset.")
            logger.debug(f"Merged DataFrame Shape: {merged_df.shape}")
        else:
            logger.info("Merged player heights into granular dataset.")

        # Analyze releases and add angles
        analyzed_df = analyze_release_all_trials(
            merged_df,
            reference_df,
            distance_to_basket=15,
            debug=debug
        )

        if debug:
            logger.debug("Final granular dataset with optimized angles created.")
            logger.debug(f"Final Granular DataFrame Shape: {analyzed_df.shape}")
        else:
            logger.info("Final granular dataset with optimized angles created.")

        return analyzed_df

    except Exception as e:
        logger.error(f"Error in add_optimized_angles_to_granular: {e}")
        return final_granular_df

def add_optimized_angles_to_ml(final_ml_df: pd.DataFrame, aggregated_angles_df: pd.DataFrame, debug: bool = False):
    """
    Merge optimized angles into the ML dataset.

    Parameters:
    - final_ml_df: DataFrame representing the ML dataset.
    - aggregated_angles_df: DataFrame with aggregated angles per trial_id.
    - debug: Whether to enable debug output.

    Returns:
    - final_ml_df_with_angles: Updated ML dataset with optimized angles.
    """
    global logger
    try:
        # Merge aggregated angles into ML dataset
        merged_df = merge_final_ml_dataset(final_ml_df, aggregated_angles_df, debug=debug)

        if debug:
            logger.debug("Final ML dataset with optimized angles created.")
            logger.debug(f"Final ML DataFrame Shape: {merged_df.shape}")
        else:
            logger.info("Merged optimized angles into ML dataset.")

        return merged_df

    except Exception as e:
        logger.error(f"Error in add_optimized_angles_to_ml: {e}")
        return final_ml_df

if __name__ == "__main__":
    try:
        # Define the debug flag
        debug = True  # Set to True for detailed logs
        # Reconfigure logging based on the debug flag
        logger = configure_logging(debug=debug)

        # Log trial IDs before processing
        log_trial_ids(final_granular_df, "Initial Load - Granular DF", debug=debug)
        log_trial_ids(final_ml_df, "Initial Load - ML DF", debug=debug)

        # Check for duplicates in final_ml_df instead of final_granular_df
        check_duplicates(final_ml_df, 'final_ml_df', debug=debug)

        # Create reference table
        reference_df = create_optimal_angle_reference_table(debug=debug)

        # Add optimized angles to granular dataset
        final_granular_df_with_optimal_release_angles = add_optimized_angles_to_granular(
            final_granular_df,
            final_ml_df,
            reference_df,
            debug=debug
        )

        # Aggregate angles (now includes angle_difference)
        aggregated_angles_df = aggregate_angles(final_granular_df_with_optimal_release_angles, debug=debug)

        # Add optimized angles to ML dataset
        final_ml_df_with_optimal_release_angles = add_optimized_angles_to_ml(
            final_ml_df,
            aggregated_angles_df,
            debug=debug
        )
        print(final_ml_df_with_optimal_release_angles)
        logger.info("All processing steps completed successfully.")

    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}")
