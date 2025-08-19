import os
import json
import pickle
import logging
import requests
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# add for if we ever want to recalculate the shap min and max values
from ..ml.config.config_loader import load_config
from ..ml.config.config_models import AppConfig
    
# ------------------------------------------------------------------------------
# Logging configuration (used by bayesian metrics functions)
# ------------------------------------------------------------------------------
logger = logging.getLogger('combined_feature_engineering')
if not logger.hasHandlers():
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def round_numeric_columns(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """
    Rounds all float columns in the DataFrame to the specified number of decimals.
    """
    float_cols = df.select_dtypes(include=['float']).columns
    df[float_cols] = df[float_cols].round(decimals)
    return df

# ------------------------------------------------------------------------------
# SHOT METER FEATURE ENGINEERING FUNCTIONS
# ------------------------------------------------------------------------------

def calculate_release_angles(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Calculate and merge release angles (mean knee, wrist, and elbow angles) based on release_point_filter.
    """
    release_df = df[df['release_point_filter'] == 1]
    if debug:
        print(f"[calculate_release_angles] release_df shape: {release_df.shape}")

    release_angles = (
        release_df.groupby('trial_id')
        .agg({
            'knee_angle': 'mean',
            'wrist_angle': 'mean',
            'elbow_angle': 'mean'
        })
        .rename(columns={
            'knee_angle': 'release_knee_angle',
            'wrist_angle': 'release_wrist_angle',
            'elbow_angle': 'release_elbow_angle'
        })
    )
    if debug:
        print(f"[calculate_release_angles] release_angles shape: {release_angles.shape}")
        print(f"[calculate_release_angles] New columns: {list(release_angles.columns)}")

    df = df.merge(release_angles, on='trial_id', how='left')
    if debug:
        print(f"[calculate_release_angles] Final df shape after merge: {df.shape}")
    else:
        print("[calculate_release_angles] Step completed.")

    return df


def calculate_optimal_release_angle_ranges(
    df: pd.DataFrame,
    debug: bool = False,
    calc_feedback_range_percentile: float = 10  # new single parameter for feedback range
) -> pd.DataFrame:
    """
    Compute initial and filtered optimal angle ranges for knee, wrist, and elbow at the release point.
    
    Instead of using two percentiles to determine the feedback range, this version uses a single 
    parameter (calc_feedback_range_percentile). For each joint, it computes the full original range 
    from the release data and then calculates a symmetric margin:
    
        margin = (orig_max - orig_min) * (calc_feedback_range_percentile / 100) / 2
    
    Then, for each joint, the filtered optimal range is defined as:
    
        filtered_optimal_min = optimal_value - margin
        filtered_optimal_max = optimal_value + margin
    
    The resulting boundaries are used for classifying each shot as 'Early', 'Late', or 'Good'.
    """
    step = "Optimal Release Angle Ranges"
    # Filter the release data (release_point_filter==1 and result==1)
    release_df = df[(df['release_point_filter'] == 1) & (df['result'] == 1)]
    if debug:
        print(f"[{step}] Filtered release_df shape: {release_df.shape}")
    else:
        print(f"[{step}] Filtering completed.")
    
    # Prepare new columns (set as NaN by default)
    new_cols = [
        'knee_release_angle_initial_optimal_min', 'knee_release_angle_initial_optimal_max',
        'wrist_release_angle_initial_optimal_min', 'wrist_release_angle_initial_optimal_max',
        'elbow_release_angle_initial_optimal_min', 'elbow_release_angle_initial_optimal_max'
    ]
    for col in new_cols:
        df[col] = float('nan')
    
    if not release_df.empty:
        # For each joint, compute the full original range and calculate margin
        # Also, retrieve the optimal release angle from the merged column (assumed constant across rows)
        # Note: The optimal value is taken from the first available row for each joint.
        # This block computes the optimal ranges for knee, wrist, and elbow.
        # -- Knee --
        orig_min_knee = release_df['knee_angle'].min()
        orig_max_knee = release_df['knee_angle'].max()
        full_range_knee = orig_max_knee - orig_min_knee
        margin_knee = full_range_knee * (calc_feedback_range_percentile / 100.0) / 2.0
        optimal_knee = df['release_knee_angle'].iloc[0]
        knee_filtered_optimal_min = optimal_knee - margin_knee
        knee_filtered_optimal_max = optimal_knee + margin_knee

        # -- Wrist --
        orig_min_wrist = release_df['wrist_angle'].min()
        orig_max_wrist = release_df['wrist_angle'].max()
        full_range_wrist = orig_max_wrist - orig_min_wrist
        margin_wrist = full_range_wrist * (calc_feedback_range_percentile / 100.0) / 2.0
        optimal_wrist = df['release_wrist_angle'].iloc[0]
        wrist_filtered_optimal_min = optimal_wrist - margin_wrist
        wrist_filtered_optimal_max = optimal_wrist + margin_wrist

        # -- Elbow --
        orig_min_elbow = release_df['elbow_angle'].min()
        orig_max_elbow = release_df['elbow_angle'].max()
        full_range_elbow = orig_max_elbow - orig_min_elbow
        margin_elbow = full_range_elbow * (calc_feedback_range_percentile / 100.0) / 2.0
        optimal_elbow = df['release_elbow_angle'].iloc[0]
        elbow_filtered_optimal_min = optimal_elbow - margin_elbow
        elbow_filtered_optimal_max = optimal_elbow + margin_elbow

        if debug:
            print(f"[{step}] Computed feedback ranges using calc_feedback_range_percentile = {calc_feedback_range_percentile}%:")
            print(f"         Knee: full_range={full_range_knee:.2f}, margin={margin_knee:.2f}, range=[{knee_filtered_optimal_min:.2f}, {knee_filtered_optimal_max:.2f}]")
            print(f"         Wrist: full_range={full_range_wrist:.2f}, margin={margin_wrist:.2f}, range=[{wrist_filtered_optimal_min:.2f}, {wrist_filtered_optimal_max:.2f}]")
            print(f"         Elbow: full_range={full_range_elbow:.2f}, margin={margin_elbow:.2f}, range=[{elbow_filtered_optimal_min:.2f}, {elbow_filtered_optimal_max:.2f}]")
        
        # Save the filtered optimal ranges into the dataframe
        df['knee_release_angle_filtered_optimal_min'] = knee_filtered_optimal_min
        df['knee_release_angle_filtered_optimal_max'] = knee_filtered_optimal_max
        df['wrist_release_angle_filtered_optimal_min'] = wrist_filtered_optimal_min
        df['wrist_release_angle_filtered_optimal_max'] = wrist_filtered_optimal_max
        df['elbow_release_angle_filtered_optimal_min'] = elbow_filtered_optimal_min
        df['elbow_release_angle_filtered_optimal_max'] = elbow_filtered_optimal_max

        # Define a helper function for classification.
        def classify_joint(angle_value, min_val, max_val):
            if angle_value < min_val:
                return "Early"
            elif angle_value > max_val:
                return "Late"
            else:
                return "Good"

        # Loop over the joints to classify the shots.
        for joint in ['knee', 'wrist', 'elbow']:
            # For release metrics, use the computed release value.
            angle_col = f"release_{joint}_angle"
            optimal_min_col = f"{joint}_release_angle_filtered_optimal_min"
            optimal_max_col = f"{joint}_release_angle_filtered_optimal_max"
            classification_col = f"{joint}_release_angle_shot_classification"
            df[classification_col] = df.apply(
                lambda row: classify_joint(row[angle_col], row[optimal_min_col], row[optimal_max_col]),
                axis=1
            )

    else:
        if debug:
            print(f"[{step}] No rows found with release_point_filter==1 and result==1.")
        else:
            print(f"[{step}] No valid rows; step completed.")
    
    return df




def calculate_optimal_max_angle_ranges(
    df: pd.DataFrame,
    output_dir: str,
    output_filename: str = "final_granular_logistic_optimized_meter_dataset.csv",
    debug: bool = False,
    calc_feedback_range_percentile: float = 10  # new single parameter for feedback range
) -> pd.DataFrame:
    """
    Calculates optimal max angle ranges (for wrist, elbow, and knee) during active shooting motion.
    
    Instead of using two percentiles to compute the range, this version uses a single parameter 
    (calc_feedback_range_percentile) to calculate a symmetric feedback range. For each angle:
    
        full_range = max(angle) - min(angle)  (from successful shots)
        feedback_diff = full_range * (calc_feedback_range_percentile / 100)
        margin = feedback_diff / 2
        
    Then, using an "optimal" value (taken as the mean of the angle for successful shots), 
    the filtered range is defined as:
    
        filtered_optimal_min = optimal_value - margin
        filtered_optimal_max = optimal_value + margin
    
    These values are then used for classification.
    """
    step = "Optimal Max Angle Ranges"
    if debug:
        print(f"[{step}] Initial df shape: {df.shape}")
        print(f"[{step}] Columns: {df.columns.to_list()}")
        total_trials_before = df['trial_id'].nunique()
        print(f"[{step}] Total trials before filtering (result): {total_trials_before}")
    else:
        print(f"[{step}] Step started.")
    
    # Filter the data for shooting motion.
    motion_df = df[df['shooting_motion'] == 1]
    if debug:
        print(f"[{step}] Motion df shape: {motion_df.shape}")
    else:
        print(f"[{step}] Shooting motion filtering completed.")
    
    # Calculate max angles per trial.
    max_angles_per_trial = (
        motion_df.groupby('trial_id')
        .agg({'wrist_angle': 'max', 'elbow_angle': 'max', 'knee_angle': 'max'})
        .reset_index()
    )
    if debug:
        print(f"[{step}] Calculated max angles per trial. Shape: {max_angles_per_trial.shape}")
    
    merged_df = motion_df.merge(
        max_angles_per_trial.rename(columns={
            'wrist_angle': 'wrist_max_angle',
            'elbow_angle': 'elbow_max_angle',
            'knee_angle': 'knee_max_angle'
        }),
        on='trial_id',
        how='left'
    )
    merged_df['is_wrist_max_angle'] = (merged_df['wrist_angle'] == merged_df['wrist_max_angle']).astype(int)
    merged_df['is_elbow_max_angle'] = (merged_df['elbow_angle'] == merged_df['elbow_max_angle']).astype(int)
    merged_df['is_knee_max_angle'] = (merged_df['knee_angle'] == merged_df['knee_max_angle']).astype(int)
    if debug:
        print(f"[{step}] Merged df shape after marking max points: {merged_df.shape}")
    
    # Filter successful shots.
    successful_shots_df = merged_df[merged_df['result'] == 1]
    if debug:
        print(f"[{step}] Successful shots df shape: {successful_shots_df.shape}")
        print(f"[{step}] Trials after result filter: {successful_shots_df['trial_id'].nunique()}")
    else:
        print(f"[{step}] Successful shots filtering completed.")
    
    # Dictionary to store computed optimal ranges.
    stats = {}
    # For each angle (using the "max" prefix), compute a symmetric feedback range.
    for joint in ['wrist', 'elbow', 'knee']:
        angle_col = f"{joint}_max_angle"
        orig_min = successful_shots_df[angle_col].min()
        orig_max = successful_shots_df[angle_col].max()
        full_range = orig_max - orig_min
        feedback_diff = full_range * (calc_feedback_range_percentile / 100.0)
        margin = feedback_diff / 2.0
        optimal_value = successful_shots_df[angle_col].mean()
        filtered_optimal_min = optimal_value - margin
        filtered_optimal_max = optimal_value + margin
        
        stats[f"{joint}_max_angle_filtered_optimal_min"] = filtered_optimal_min
        stats[f"{joint}_max_angle_filtered_optimal_max"] = filtered_optimal_max
        
        if debug:
            print(f"[{step}] For {angle_col}: full_range={full_range:.2f}, feedback_diff={feedback_diff:.2f}, margin={margin:.2f}, optimal={optimal_value:.2f}, range=[{filtered_optimal_min:.2f}, {filtered_optimal_max:.2f}]")
    
    # Save the computed stats into the merged_df as constants.
    for key, value in stats.items():
        merged_df[key] = value
    
    # Define a helper function for classification.
    def classify_joint(angle_value, min_val, max_val):
        if angle_value < min_val:
            return "Early"
        elif angle_value > max_val:
            return "Late"
        else:
            return "Good"
    
    # Loop over the joints to classify the shots based on max angles.
    for joint in ['wrist', 'elbow', 'knee']:
        classification_col = f"{joint}_max_angle_shot_classification"
        optimal_min = stats[f"{joint}_max_angle_filtered_optimal_min"]
        optimal_max = stats[f"{joint}_max_angle_filtered_optimal_max"]
        # Use the original angle column (e.g., 'wrist_angle') for classification.
        merged_df[classification_col] = merged_df.apply(
            lambda row, min_val=optimal_min, max_val=optimal_max, joint=joint: classify_joint(row[f"{joint}_max_angle"], min_val, max_val),
            axis=1
        )
    
    merged_df = round_numeric_columns(merged_df, decimals=2)
    
    output_path = os.path.join(output_dir, output_filename)
    merged_df.to_csv(output_path, index=False)
    if debug:
        new_cols = list(stats.keys()) + [
            'wrist_max_angle_shot_classification',
            'elbow_max_angle_shot_classification',
            'knee_max_angle_shot_classification',
            'is_wrist_max_angle', 'is_elbow_max_angle', 'is_knee_max_angle'
        ]
        print(f"[{step}] Updated dataset saved to {output_path}")
        print(f"[{step}] New columns added: {new_cols}")
        print(f"[{step}] Data types of new columns:")
        print(merged_df[new_cols].dtypes)
    else:
        print(f"[{step}] Step completed.")
    
    return merged_df




def load_features_from_pickle(pickle_path: str, y_variable: str = 'result') -> List[str]:
    """
    Load features from a pickle file (list or DataFrame) and remove the y variable.
    """
    try:
        if pickle_path.startswith('http://') or pickle_path.startswith('https://'):
            response = requests.get(pickle_path)
            response.raise_for_status()
            pickle_file = BytesIO(response.content)
            features_data = pickle.load(pickle_file)
        else:
            with open(pickle_path, 'rb') as f:
                features_data = pickle.load(f)
        if isinstance(features_data, pd.DataFrame):
            features = list(features_data.columns)
        elif isinstance(features_data, list):
            features = features_data
        else:
            raise ValueError("Pickle file must contain a list or a DataFrame of features.")
        if y_variable in features:
            features.remove(y_variable)
        logger.debug(f"[load_features_from_pickle] Loaded features: {features}")
        return features
    except Exception as e:
        logger.error(f"[load_features_from_pickle] Error: {e}")
        raise


def load_precalculated_bayesian_metrics(
    bayesian_metrics_file_path: str,
    debug: bool = False
) -> pd.DataFrame:
    """
    Load precalculated bayesian metrics from a CSV file and rename the columns.
    """
    try:
        if debug:
            logger.debug(f"[load_precalculated_bayesian_metrics] Loading from {bayesian_metrics_file_path}...")
        df = pd.read_csv(bayesian_metrics_file_path)
        required_columns = [
            "Parameter",
            "Optimized (Candidate, Real)",
            "Baseline (Real)",
            "Min (Real)",
            "Max (Real)",
            "Success Rate (Baseline)",
            "Success Rate (Candidate)"
        ]
        df = df[required_columns]
        df.rename(
            columns={
                "Optimized (Candidate, Real)": "bayes_optimized",
                "Baseline (Real)": "baseline",
                "Min (Real)": "bayes_min",
                "Max (Real)": "bayes_max",
                "Success Rate (Baseline)": "baseline_success_rate",
                "Success Rate (Candidate)": "bayes_success_rate"
            },
            inplace=True
        )
        if debug:
            logger.debug(f"[load_precalculated_bayesian_metrics] DataFrame shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"[load_precalculated_bayesian_metrics] Error: {e}")
        raise


def add_bayesian_ranges_to_ml_dataset(
    bayesian_metrics_data: pd.DataFrame,
    final_ml_data_path: str,
    output_dir: str,
    bayes_min_max_range_percentile: float,       # new parameter, e.g., 10 for 10%
    output_filename: str = "ml_dataset_with_bayesian_metrics.csv",
    debug: bool = False
) -> pd.DataFrame:
    """
    Append precalculated bayesian metrics (optimized, min, max) to the final ML dataset.

    In this updated version, instead of basing bayes_min and bayes_max on quantiles
    of the metric values, we use the full original range of the metric (bayes_orig_min and bayes_orig_max)
    to determine a symmetric range. We calculate the difference as:
    
        bayes_min_max_diff = (bayes_orig_max - bayes_orig_min) * (bayes_min_max_range_percentile / 100)
    
    and define the margin as half of that difference. The bayesian range is then defined as:
    
        bayes_min = bayes_optimized - margin
        bayes_max = bayes_optimized + margin
    
    Additionally, we add a constraint so that bayes_min cannot be lower than bayes_orig_min and
    bayes_max cannot be higher than bayes_orig_max.
    
    The original min and max values are stored as bayes_orig_min and bayes_orig_max.
    """
    step = "Add Bayesian Ranges to ML Dataset"
    try:
        # Map metric name (lowercase) to its bayes_optimized value.
        opt_dict = pd.Series(
            bayesian_metrics_data['bayes_optimized'].values,
            index=bayesian_metrics_data['Parameter'].str.lower()
        ).to_dict()
        
        if debug:
            logger.debug(f"[{step}] Loading final ML dataset from {final_ml_data_path}...")
        final_ml_data = pd.read_csv(final_ml_data_path)
        
        for metric, opt_value in opt_dict.items():
            optimized_col = f"{metric}_bayes_optimized"
            # Add the bayes_optimized value as a new column.
            final_ml_data[optimized_col] = opt_value
            if debug:
                logger.debug(f"[{step}] Added column '{optimized_col}' with value: {opt_value}")
            
            if metric in final_ml_data.columns:
                # Retrieve bayes_optimized value (assumed constant).
                bayes_opt_val = final_ml_data[optimized_col].iloc[0]
                
                # Obtain the original full range of the metric.
                orig_min = final_ml_data[metric].min()
                orig_max = final_ml_data[metric].max()
                full_range = orig_max - orig_min
                
                # Compute the bayes_min_max difference based on the new parameter.
                bayes_min_max_diff = full_range * (bayes_min_max_range_percentile / 100.0)
                # The margin is half of that difference.
                margin = bayes_min_max_diff / 2.0
                
                # Define the new bayesian range centered on bayes_opt_val.
                new_min = bayes_opt_val - margin
                new_max = bayes_opt_val + margin
                
                # Constrain the range so it does not exceed the original bounds.
                if new_min < orig_min:
                    new_min = orig_min
                if new_max > orig_max:
                    new_max = orig_max
                
                # Save the computed bayesian range.
                final_ml_data[f"{metric}_bayes_min"] = new_min
                final_ml_data[f"{metric}_bayes_max"] = new_max
                if debug:
                    logger.debug(f"[{step}] For metric '{metric}':")
                    logger.debug(f"         bayes_opt_val: {bayes_opt_val}")
                    logger.debug(f"         orig_min: {orig_min}, orig_max: {orig_max} (full_range: {full_range})")
                    logger.debug(f"         bayes_min_max_diff (full_range * {bayes_min_max_range_percentile}%): {bayes_min_max_diff}")
                    logger.debug(f"         margin (each side): {margin}")
                    logger.debug(f"         Set bayes_min: {new_min}, bayes_max: {new_max}")
                
                # Also store the original min and max values.
                final_ml_data[f"{metric}_bayes_orig_min"] = orig_min
                final_ml_data[f"{metric}_bayes_orig_max"] = orig_max
            else:
                if debug:
                    logger.warning(f"[{step}] Base metric column '{metric}' not found in final ML data; skipping bayesian range computation.")
        
        # Round numeric columns.
        final_ml_data = round_numeric_columns(final_ml_data, decimals=2)
        
        output_path = os.path.join(output_dir, output_filename)
        final_ml_data.to_csv(output_path, index=False)
        if debug:
            logger.debug(f"[{step}] Updated ML dataset saved to {output_path}")
        else:
            print(f"[{step}] Step completed.")
        return final_ml_data
    except Exception as e:
        logger.error(f"[{step}] Error: {e}")
        raise




def classify_metrics(
    final_ml_data: pd.DataFrame,
    bayesian_metrics: List[str],
    output_dir: str,
    output_filename: str = "classified_ml_dataset.csv",
    debug: bool = False
) -> pd.DataFrame:
    """
    Classify each bayesian metric in the ML dataset as 'Early', 'Late', or 'Good'
    based on the corresponding bayesian min/max values.
    
    This updated version accumulates new columns in a dictionary and then concatenates
    them all at once to reduce DataFrame fragmentation.
    """
    step = "Classify Metrics"
    try:
        # Create a dictionary to hold all new columns for batch addition.
        new_cols = {}
        
        # Loop over each metric and compute the required new columns.
        for metric in bayesian_metrics:
            base_metric_col = metric
            bayes_min_col = f"{metric}_bayes_min"
            bayes_max_col = f"{metric}_bayes_max"
            classification_col = f"{metric}_bayes_classification"
            
            # Skip metric if required bayesian range columns are missing.
            if bayes_min_col not in final_ml_data.columns or bayes_max_col not in final_ml_data.columns:
                if debug:
                    logger.warning(f"[{step}] Bayesian range columns for '{metric}' not found. Skipping classification.")
                continue
            
            # Compute the classification column using np.select.
            new_cols[classification_col] = np.select(
                [
                    final_ml_data[base_metric_col] < final_ml_data[bayes_min_col],
                    final_ml_data[base_metric_col] > final_ml_data[bayes_max_col]
                ],
                ['Early', 'Late'],
                default='Good'
            )
            
            # Compute the unit change column if the bayes optimized column exists.
            bayes_optimized_col = f"{metric}_bayes_optimized"
            if bayes_optimized_col in final_ml_data.columns:
                new_cols[f"{metric}_bayes_unit_change"] = final_ml_data[base_metric_col] - final_ml_data[bayes_optimized_col]
                if debug:
                    logger.debug(f"[{step}] Prepared column '{metric}_bayes_unit_change' computed as {base_metric_col} - {bayes_optimized_col}.")
            else:
                if debug:
                    logger.warning(f"[{step}] Bayes optimized column '{bayes_optimized_col}' not found for metric '{metric}'.")
        
        # After processing all metrics, create a DataFrame from the new columns dictionary.
        new_cols_df = pd.DataFrame(new_cols, index=final_ml_data.index)
        
        # Concatenate the new columns with the original DataFrame.
        final_ml_data = pd.concat([final_ml_data, new_cols_df], axis=1)
        
        # Use copy() to defragment the DataFrame.
        final_ml_data = final_ml_data.copy()
        
        if debug:
            logger.debug(f"[{step}] Classification and bayesian unit change computation completed for all metrics.")
        
        # Round numeric columns as before.
        final_ml_data = round_numeric_columns(final_ml_data, decimals=2)
        
        # Save the updated dataset.
        output_path = os.path.join(output_dir, output_filename)
        final_ml_data.to_csv(output_path, index=False)
        if debug:
            logger.debug(f"[{step}] Classified dataset saved to {output_path}")
        else:
            print(f"[{step}] Step completed.")
        
        return final_ml_data
    except Exception as e:
        logger.error(f"[{step}] Error: {e}")
        raise



def merge_bayes_metrics_with_granular_data(
    granular_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    bayesian_metrics: List[str],
    output_dir: str,
    output_filename: str = "bayesian_shot_meter_granular_dataset.csv",
    debug: bool = False
) -> pd.DataFrame:
    """
    Merge the ML dataset (with bayesian metrics) with the granular dataset on 'trial_id'.
    """
    step = "Merge Bayesian Metrics with Granular Data"
    try:
        if debug:
            logger.debug(f"[{step}] Granular df trial_id count: {granular_df['trial_id'].nunique()}")
            logger.debug(f"[{step}] ML df trial_id count: {ml_df['trial_id'].nunique()}")
            logger.debug(f"[{step}] Performing left join on 'trial_id'...")
        merged_dataset = granular_df.merge(ml_df, on='trial_id', how='left', suffixes=('', '_meter'))
        merged_dataset = round_numeric_columns(merged_dataset, decimals=2)
        
        output_path = os.path.join(output_dir, output_filename)
        merged_dataset.to_csv(output_path, index=False)
        if debug:
            logger.debug(f"[{step}] Merged dataset shape: {merged_dataset.shape}")
            logger.debug(f"[{step}] Merged dataset saved to {output_path}")
        else:
            print(f"[{step}] Step completed.")
        return merged_dataset
    except Exception as e:
        logger.error(f"[{step}] Error: {e}")
        raise


def bayesian_optimized_granular_data_main(
    bayesian_metrics_file_path: str = "../../data/predictions/bayesian_optimization_results/bayesian_optimization_results.csv",
    final_ml_file_path: str = "../../data/processed/final_ml_dataset.csv",
    final_ml_with_predictions_path: str = "../../data/model/predictions/final_dataset_with_predictions_and_shap.csv",
    final_granular_file_path: str = "../../data/processed/final_granular_dataset.csv",
    pickle_path: str = "../../data/preprocessor/features_info/final_ml_df_selected_features_columns.pkl",
    y_variable: str = "result",
    output_dir: str = "../../data/model/shot_meter_docs/",
    debug: bool = False,
    bayes_min_max_range_percentile: float = 10, 
    output_filename: str = "bayesian_shot_meter_granular_dataset.csv"
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Main function for bayesian optimization processing. Loads feature names,
    bayesian metrics, adds these metrics to the ML dataset (with new bayesian min/max computed using the provided percentiles),
    performs classification, and finally merges with the granular dataset.
    """
    step = "Bayesian Optimized Granular Data Main"
    try:
        if debug:
            logger.debug(f"[{step}] Starting main function.")

        # Step 0: Load feature names from pickle
        if debug:
            logger.debug(f"[{step}] Loading feature names from: {pickle_path}")
        bayesian_metrics = load_features_from_pickle(pickle_path=pickle_path, y_variable=y_variable)
        if debug:
            logger.debug(f"[{step}] Loaded bayesian metrics: {bayesian_metrics}")

        # Step 1: Load precalculated bayesian metrics table
        combined_data = load_precalculated_bayesian_metrics(
            bayesian_metrics_file_path=bayesian_metrics_file_path,
            debug=debug
        )

        # Step 2: Add bayesian ranges to the ML dataset.
        final_ml_bayes_df = add_bayesian_ranges_to_ml_dataset(
            bayesian_metrics_data=combined_data,
            final_ml_data_path=final_ml_with_predictions_path,
            output_dir=output_dir,
            bayes_min_max_range_percentile=bayes_min_max_range_percentile,
            debug=debug
        )

        # Step 3: Classify metrics in the ML dataset
        final_ml_bayes_df = classify_metrics(
            final_ml_data=final_ml_bayes_df,
            bayesian_metrics=bayesian_metrics,
            output_dir=output_dir,
            debug=debug
        )

        # (Optional) Debug: Load original ML dataset to check columns
        final_ml_data = pd.read_csv(final_ml_with_predictions_path)
        if debug:
            logger.debug(f"[{step}] Final ML dataset columns: {final_ml_data.columns.to_list()}")

        # Step 4: Merge the ML dataset with the granular dataset
        granular_df = pd.read_csv(final_granular_file_path)
        merged_data = merge_bayes_metrics_with_granular_data(
            granular_df=granular_df,
            ml_df=final_ml_bayes_df,
            bayesian_metrics=bayesian_metrics,
            output_dir=output_dir,
            debug=debug,
            output_filename=output_filename
        )

        # Step 5: Build the bayesian_metrics_dict for downstream use
        bayesian_metrics_dict = {}
        for metric in bayesian_metrics:
            optimized_metric_col = f"{metric}_bayes_optimized"
            bayes_classification_col = f"{metric}_bayes_classification"
            shap_importance_col = f"shap_{metric}_importance"
            min_col = f"{metric}_bayes_min"
            max_col = f"{metric}_bayes_max"
            orig_min_col = f"{metric}_bayes_orig_min"
            orig_max_col = f"{metric}_bayes_orig_max"
            if optimized_metric_col in merged_data.columns:
                bayesian_metrics_dict[metric] = {
                    'bayes_optimal': merged_data.at[0, optimized_metric_col],
                    'bayes_original': merged_data.at[0, metric],
                    'bayes_min': merged_data.at[0, min_col] if min_col in merged_data.columns else None,
                    'bayes_max': merged_data.at[0, max_col] if max_col in merged_data.columns else None,
                    'bayes_orig_min': merged_data.at[0, orig_min_col] if orig_min_col in merged_data.columns else None,
                    'bayes_orig_max': merged_data.at[0, orig_max_col] if orig_max_col in merged_data.columns else None,
                    'bayes_classification': merged_data.at[0, bayes_classification_col] if bayes_classification_col in merged_data.columns else "Good",
                    'shap_importance': merged_data.at[0, shap_importance_col] if shap_importance_col in merged_data.columns else None
                }
            else:
                if debug:
                    logger.warning(f"[{step}] Optimized metric column for '{metric}' not found. Skipping.")
        no_change_list = ["calculated_release_angle", "release_angle"]
        for metric, values in bayesian_metrics_dict.items():
            filter_name = metric if metric in no_change_list else metric.replace("release_", "").replace("_max", "")
            bayesian_metrics_dict[metric]['filter_name'] = filter_name

        for metric in bayesian_metrics:
            shap_direction_col = f"shap_{metric}_direction"
            if shap_direction_col in merged_data.columns:
                bayesian_metrics_dict[metric]['shap_direction'] = merged_data.at[0, shap_direction_col]
                if debug:
                    logger.debug(f"[{step}] Extracted '{shap_direction_col}': {merged_data.at[0, shap_direction_col]}")
            else:
                if debug:
                    logger.warning(f"[{step}] Missing shap_direction column '{shap_direction_col}' for metric '{metric}'.")
                bayesian_metrics_dict[metric]['shap_direction'] = None

        if debug:
            logger.debug(f"[{step}] Bayesian Metrics Dictionary:\n{json.dumps(bayesian_metrics_dict, indent=4)}")

        bayesian_metrics_output_path = os.path.join(output_dir, "bayesian_metrics_dict.json")
        with open(bayesian_metrics_output_path, 'w') as f:
            json.dump(bayesian_metrics_dict, f, indent=4)
        if debug:
            logger.debug(f"[{step}] Bayesian metrics dictionary saved to {bayesian_metrics_output_path}")
        else:
            print(f"[{step}] Step completed.")

        return merged_data, bayesian_metrics_dict

    except Exception as e:
        logger.error(f"[{step}] Error: {e}")
        raise


def automated_bayes_shap_summary(
    granular_data_path: str,
    release_angles_output_dir: str,
    max_angles_output_dir: str,
    bayesian_metrics_file_path: str,
    final_ml_file_path: str,
    final_ml_with_predictions_path: str,
    pickle_path: str,
    y_variable: str,
    bayes_min_max_range_percentile: float,
    calc_feedback_range_percentile: float,
    output_dir: str,
    output_filename: str,
    debug: bool = False,
    # New parameters for reloading SHAP predictions:
    reload_shap_data: bool = False,
    new_shap_min_max_percentile: Optional[float] = None,
    config: Optional["AppConfig"] = None,  # Make sure to import AppConfig from your config module
    streamlit_app_paths: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, Dict[str, int]]]:
    """
    Automated summary function that runs the entire data merging and bayesian metrics
    processing pipeline. It:
      1. Loads the granular dataset.
      2. Calculates release angles and optimal release angle ranges using the specified feedback range.
      3. Saves an intermediate dataset (optional).
      4. Calculates optimal max angle ranges using the specified feedback range.
      5. Optionally reloads the SHAP predictions (via predict_and_shap) using a new metrics_percentile.
      6. Runs the bayesian optimization merging and classification.
      7. Summarizes the bayesian classification results (counts of 'Good', 'Early', 'Late')
         for each metric.
    
    New Parameters:
      - reload_shap_data: If True, re-run the predict_and_shap module to recalculate SHAP columns.
      - new_shap_min_max_percentile: If provided (and reload_shap_data is True), use this percentile when
          recalculating SHAP feedback.
      - config: An AppConfig object required to run predict_and_shap. If not provided, you must load it before calling.
    
    Returns:
      A tuple containing:
        - The final merged DataFrame.
        - The bayesian metrics dictionary.
        - A summary dictionary with counts for each bayes_classification per metric.
    """
    # 1. Load granular dataset
    if debug:
        print("[automated_bayes_shap_summary] Loading granular dataset...")
    df_granular = pd.read_csv(granular_data_path)
    
    # 2. Calculate release angles and optimal release angle ranges.
    if debug:
        print("[automated_bayes_shap_summary] Calculating release angles...")
    df_granular = calculate_release_angles(df_granular, debug=debug)
    if debug:
        print("[automated_bayes_shap_summary] Calculating optimal release angle ranges...")
    df_granular = calculate_optimal_release_angle_ranges(
        df_granular,
        debug=debug,
        calc_feedback_range_percentile=calc_feedback_range_percentile
    )
    # Optionally save the intermediate release angles dataset.
    release_angles_output_path = os.path.join(release_angles_output_dir, "granular_with_release_angles.csv")
    df_granular.to_csv(release_angles_output_path, index=False)
    if debug:
        print(f"[automated_bayes_shap_summary] Granular dataset with release angles saved to {release_angles_output_path}")
    
    # 3. Calculate optimal max angle ranges.
    if debug:
        print("[automated_bayes_shap_summary] Calculating optimal max angle ranges...")
    max_angles_output_filename = "final_granular_logistic_optimized_meter_dataset.csv"
    df_granular = calculate_optimal_max_angle_ranges(
        df_granular,
        output_dir=max_angles_output_dir,
        output_filename=max_angles_output_filename,
        debug=debug,
        calc_feedback_range_percentile=calc_feedback_range_percentile
    )
    
    # 4. Optionally, if requested, reload the SHAP predictions with an updated percentile.
    #    This re-runs the predict_and_shap module so that the final ML dataset (with SHAP columns)
    #    reflects the new metrics_percentile.
    if reload_shap_data:
        if config is None:
            raise ValueError("To reload SHAP data, you must provide a valid AppConfig via the 'config' parameter.")
        # Use the new_shap_min_max_percentile if provided; otherwise, default to the original one.
        reload_percentile = new_shap_min_max_percentile if new_shap_min_max_percentile is not None else 10.0
        if debug:
            print(f"[automated_bayes_shap_summary] Reloading SHAP data with metrics_percentile = {reload_percentile}")
        # Here we assume that the raw data for prediction is available.
        # For example, you might reload the raw dataset (or use the one already processed).
        # In this example, we assume raw data is available at granular_data_path (adjust if needed).
        df_predict = pd.read_csv(final_ml_file_path)
        
        if streamlit_app_paths:
            ordinal_file=Path('data/preprocessor/features_info/ordinal_categoricals.pkl')
            nominal_file=Path('data/preprocessor/features_info/nominal_categoricals.pkl')
            numericals_file=Path('data/preprocessor/features_info/numericals.pkl')
            y_variable_file=Path('data/preprocessor/features_info/y_variable.pkl')
        else:
            ordinal_file=Path('../../data/preprocessor/features_info/ordinal_categoricals.pkl')
            nominal_file=Path('../../data/preprocessor/features_info/nominal_categoricals.pkl')
            numericals_file=Path('../../data/preprocessor/features_info/numericals.pkl')
            y_variable_file=Path('../../data/preprocessor/features_info/y_variable.pkl')
        # We choose a temporary output directory for the reloaded SHAP data.
        predictions_output_path = Path(config.paths.predictions_output_dir).resolve() / "shap_results"
        model_output_dir = predictions_output_path 
        shap_results = predict_and_shap(
            config=config,
            df_input=df_predict,
            save_dir=model_output_dir,
            columns_to_add=['trial_id'],
            generate_summary_plot=True,
            generate_dependence_plots=True,
            generate_force_plots=True,
            force_plot_indices=[0],
            top_n_features=10,
            use_mad=False,
            logger=logger,
            features_file=(Path(config.paths.data_dir) / config.paths.features_metadata_file).resolve(),
            ordinal_file=ordinal_file,
            nominal_file=nominal_file,
            numericals_file=numericals_file,
            y_variable_file=y_variable_file,
            model_save_dir_override=Path(config.paths.model_save_base_dir),
            transformers_dir_override=Path(config.paths.transformers_save_base_dir),
            metrics_percentile=reload_percentile
        )

    # 5. Run bayesian optimized granular data processing.
    merged_data, bayesian_metrics_dict = bayesian_optimized_granular_data_main(
        bayesian_metrics_file_path=bayesian_metrics_file_path,
        final_ml_file_path=final_ml_file_path,
        final_ml_with_predictions_path=final_ml_with_predictions_path,
        final_granular_file_path=os.path.join(max_angles_output_dir, max_angles_output_filename),
        pickle_path=pickle_path,
        y_variable=y_variable,
        output_dir=output_dir,
        debug=debug,
        bayes_min_max_range_percentile=bayes_min_max_range_percentile,
        output_filename=output_filename
    )
    
    # 6. Summarize bayesian classification results: count how many 'Good', 'Early', 'Late' per metric.
    classification_summary = {}
    for metric in bayesian_metrics_dict.keys():
        col = f"{metric}_bayes_classification"
        if col in merged_data.columns:
            classification_summary[metric] = merged_data[col].value_counts().to_dict()
        else:
            classification_summary[metric] = {"Error": "Column not found"}
    
    return merged_data, bayesian_metrics_dict, classification_summary




# ------------------------------------------------------------------------------
# Main Block (Example usage)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # # ------------------------------------------------------------------------------
    # # Import configuration loader and models

    # # Predict with shap so we can adjust the percentile if we ever want to
    # # Main testing block for trying multiple models.
    config_path = Path('../../data/model/preprocessor_config/preprocessor_config.yaml')
    try:
        config: AppConfig = load_config(config_path)
        print(f"Configuration loaded successfully from {config_path}.")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        exit(1)
        
    # ------------------------------------------------------------------------------
    # Bayesian Shap calc merge    
    # Example file paths and parameters (adjust these as needed)
    granular_data_path = "../../data/processed/final_granular_dataset.csv"
    release_angles_output_dir = "../../data/model/shot_meter_docs/"
    max_angles_output_dir = "../../data/model/shot_meter_docs/"
    bayesian_metrics_file_path = "../../data/predictions/bayesian_optimization_results/bayesian_optimization_results.csv"
    final_ml_file_path = "../../data/processed/final_ml_dataset.csv"
    final_ml_with_predictions_path = "../../data/predictions/shap_results/final_predictions_with_shap.csv"
    pickle_path = "../../data/preprocessor/features_info/final_ml_df_selected_features_columns.pkl"
    y_variable = "result"
    bayes_min_max_range_percentile = 10
    calc_feedback_range_percentile = 10
    new_shap_min_max_percentile = 10
    output_dir = "../../data/model/shot_meter_docs/"
    output_filename = "bayesian_shot_meter_granular_dataset.csv"
    debug = True
    
    # Run the automated summary function
    merged_data, bayesian_metrics_dict, classification_summary = automated_bayes_shap_summary(
        granular_data_path=granular_data_path,
        release_angles_output_dir=release_angles_output_dir,
        max_angles_output_dir=max_angles_output_dir,
        bayesian_metrics_file_path=bayesian_metrics_file_path,
        final_ml_file_path=final_ml_file_path,
        final_ml_with_predictions_path=final_ml_with_predictions_path,
        pickle_path=pickle_path,
        y_variable=y_variable,
        bayes_min_max_range_percentile=bayes_min_max_range_percentile,
        output_dir=output_dir,
        output_filename=output_filename,
        debug=debug,
        calc_feedback_range_percentile=calc_feedback_range_percentile,
        #Only needed if we want to reload the shap data
        reload_shap_data=True,
        new_shap_min_max_percentile=new_shap_min_max_percentile,
        config=config
    )
    
    # Print out a summary of bayesian classifications
    print("Bayesian Classification Summary (counts per metric):")
    for metric, counts in classification_summary.items():
        print(f"{metric}: {counts}")
