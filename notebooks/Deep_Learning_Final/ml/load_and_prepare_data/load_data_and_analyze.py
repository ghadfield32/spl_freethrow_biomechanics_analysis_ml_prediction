
import numpy as np
import pandas as pd
import json
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import shap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


###############################################################################
# HELPER FUNCTION FOR DEBUG OUTPUTS
###############################################################################
def _print_debug_info(step_name, df, new_columns=None, debug=False):
    """
    Prints debug information about a DataFrame after a processing step.
    
    When debug=True, prints:
      - The step name.
      - The DataFrame shape.
      - If new_columns is provided (a list of column names), prints for each:
          • Data type and a sample of unique values (up to 5).
    When debug=False, prints a single-line message indicating step completion.
    """
    if debug:
        logging.info(f"Step [{step_name}]: DataFrame shape = {df.shape}")
        if new_columns:
            logging.info(f"New columns added: {new_columns}")
            for col in new_columns:
                sample = df[col].dropna().unique()[:5]
                logging.info(f" - {col}: dtype={df[col].dtype}, sample values={sample}")
    else:
        logging.info(f"Step [{step_name}] completed.")


###############################################################################
# FUNCTION DEFINITIONS
###############################################################################
def load_data(csv_path, json_path, participant_id='P0001', debug=False):
    """
    Loads the main dataset and participant information, then merges them.
    
    Parameters:
      - csv_path (str): Path to the main CSV file.
      - json_path (str): Path to the participant information JSON file.
      - participant_id (str): Participant identifier.
      - debug (bool): If True, prints detailed debug info.
    
    Returns:
      - data (pd.DataFrame): Merged DataFrame.
    """
    try:
        data = pd.read_csv(csv_path)
        logging.info(f"Loaded data from {csv_path} with shape {data.shape}")
    except FileNotFoundError:
        logging.error(f"File not found: {csv_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading {csv_path}: {e}")
        sys.exit(1)
    
    data['participant_id'] = participant_id
    logging.info(f"Added 'participant_id' column with value '{participant_id}'")
    
    try:
        with open(json_path, 'r') as file:
            participant_info = json.load(file)
        participant_df = pd.DataFrame([participant_info])
        logging.info(f"Loaded participant information from {json_path}")
    except FileNotFoundError:
        logging.error(f"File not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in {json_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading {json_path}: {e}")
        sys.exit(1)
    
    data = pd.merge(data, participant_df, on='participant_id', how='left')
    logging.info(f"Merged participant data. New shape: {data.shape}")
    _print_debug_info("load_data", data, debug=debug)
    
    # remove these columns
    data = data.drop(columns=['event_idx_leg', 'event_idx_elbow', 'event_idx_release', 'event_idx_wrist'])
    # filter out null shooting_phases
    if 'shooting_phases' in data.columns:
        data = data[data['shooting_phases'].notnull()]

    return data



def calculate_joint_angles(df, connections, debug=False):
    """
    Calculates joint angles from coordinate data using vector mathematics.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing joint coordinates.
        connections (list): Joint connections defining biomechanical segments.
        debug (bool): Enable debug logging.
        
    Returns:
        df (pd.DataFrame): Updated DataFrame with new angle columns.
    """
    angle_columns = []
    
    # Define angle calculation points for key joints
    # Note: The new "KNEE" definition uses hip, knee, and ankle as the points.
    angle_definitions = {
        'SHOULDER': {
            'left': ['L_HIP', 'L_SHOULDER', 'L_ELBOW'],
            'right': ['R_HIP', 'R_SHOULDER', 'R_ELBOW']
        },
        'HIP': {
            'left': ['L_SHOULDER', 'L_HIP', 'L_KNEE'],
            'right': ['R_SHOULDER', 'R_HIP', 'R_KNEE']
        },
        'KNEE': {
            'left': ['L_HIP', 'L_KNEE', 'L_ANKLE'],
            'right': ['R_HIP', 'R_KNEE', 'R_ANKLE']
        },
        'ANKLE': {
            'left': ['L_KNEE', 'L_ANKLE', 'L_5THTOE'],
            'right': ['R_KNEE', 'R_ANKLE', 'R_5THTOE']
        }
    }

    for joint, sides in angle_definitions.items():
        for side in ['left', 'right']:
            points = sides[side]
            prefix = 'L' if side == 'left' else 'R'
            
            # Build list of required coordinate columns for this calculation
            required_cols = []
            for point in points:
                required_cols += [f'{point}_x', f'{point}_y', f'{point}_z']
                
            if all(col in df.columns for col in required_cols):
                # Calculate the vectors needed for the angle
                vec1 = df[[f'{points[0]}_x', f'{points[0]}_y', f'{points[0]}_z']].values - \
                       df[[f'{points[1]}_x', f'{points[1]}_y', f'{points[1]}_z']].values
                vec2 = df[[f'{points[2]}_x', f'{points[2]}_y', f'{points[2]}_z']].values - \
                       df[[f'{points[1]}_x', f'{points[1]}_y', f'{points[1]}_z']].values

                # Compute the dot product and the norms of the vectors
                dot_product = np.sum(vec1 * vec2, axis=1)
                norm_product = np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1)
                
                # Compute the angle (in degrees) and add a small epsilon to avoid division by zero
                angles = np.degrees(np.arccos(dot_product / (norm_product + 1e-8)))
                
                col_name = f'{prefix}_{joint}_angle'
                df[col_name] = angles
                angle_columns.append(col_name)
                
                if debug:
                    logging.info(f"Calculated {col_name} with mean: {angles.mean():.2f}°")
            else:
                logging.warning(f"Missing coordinates for {prefix}_{joint} angle calculation")

    _print_debug_info("calculate_joint_angles", df, new_columns=angle_columns, debug=debug)
    return df



def prepare_joint_features(data, debug=False, group_trial=False, group_shot_phase=False):
    """
    Aggregates joint-level energy and power, creates additional biomechanical features,
    and adds new features:
      - energy_acceleration: instantaneous rate of change of joint_energy.
      - ankle_power_ratio: ratio of left to right ankle ongoing power.
      - Additional asymmetry metrics.
      - Power ratios for all joint pairs.
      - Side-Specific Range-of-Motion (ROM) metrics (ROM, deviation, and binary extreme flag).
      - Removal of the wrist_angle_release column if present.
      
    NEW GROUPING FEATURE (optional):
      - If group_trial is True, computes aggregated trial-level features (e.g., trial mean exhaustion and total joint energy)
        and merges them into the DataFrame.
      - If group_shot_phase is True, and if the column 'shooting_phases' exists, computes shot-phase level aggregates
        (based on both 'trial_id' and 'shooting_phases') and merges them in.
    
    Parameters:
      - data (pd.DataFrame): Input DataFrame.
      - debug (bool): If True, prints detailed debug outputs.
      - group_trial (bool): If True, add trial-level aggregated features.
      - group_shot_phase (bool): If True, add shot-phase-level aggregated features.
    
    Returns:
      - data (pd.DataFrame): Updated DataFrame with new features and, optionally, grouping-based extra columns.
    """
    step = "prepare_joint_features"
    new_cols = []
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
    # Compute joint angles first.
    data = calculate_joint_angles(data, connections, debug=debug)
    # Example: Assume a base datetime for when the recording started.
    base_datetime = pd.Timestamp('2025-01-01 00:00:00')

    # Determine the unit of continuous_frame_time:
    # If continuous_frame_time is in milliseconds, then:
    data['datetime'] = base_datetime + pd.to_timedelta(data['continuous_frame_time'], unit='ms')

    # Rename participant anthropometrics if available.
    if 'height_in_meters' in data.columns and 'weight__in_kg' in data.columns:
        data['player_height_in_meters'] = data['height_in_meters']
        data['player_weight__in_kg'] = data['weight__in_kg']
        data.drop(['height_in_meters', 'weight__in_kg'], axis=1, inplace=True, errors='ignore')
        new_cols.extend(['player_height_in_meters', 'player_weight__in_kg'])
        logging.info("Renamed participant anthropometrics.")
    else:
        logging.warning("Participant anthropometric columns not found during renaming.")

    # Identify joint energy and power columns.
    joint_energy_columns = [col for col in data.columns if '_energy' in col and not ('by_trial' in col or 'overall' in col)]
    print("Joint energy columns: ", joint_energy_columns)
    joint_power_columns = [col for col in data.columns if '_ongoing_power' in col]
    print("Joint power columns: ", joint_power_columns)
    print("All angle columns: ", [col for col in data.columns if 'angle' in col])
    logging.info(f"Identified {len(joint_energy_columns)} joint energy and {len(joint_power_columns)} joint power columns.")
    if not joint_energy_columns:
        logging.error("No joint energy columns found. Check naming conventions.")
        sys.exit(1)
    if not joint_power_columns:
        logging.error("No joint power columns found. Check naming conventions.")
        sys.exit(1)
    
    # Create aggregated columns.
    data['joint_energy'] = data[joint_energy_columns].sum(axis=1)
    data['joint_power'] = data[joint_power_columns].sum(axis=1)
    new_cols.extend(['joint_energy', 'joint_power'])
    logging.info("Created aggregated 'joint_energy' and 'joint_power'.")

    # --- NEW FEATURE: Energy Acceleration ---
    if 'continuous_frame_time' in data.columns:
        time_diff = data['continuous_frame_time'].diff().replace(0, 1e-6)  # Avoid division by zero
        data['energy_acceleration'] = data['joint_energy'].diff() / time_diff
        data['energy_acceleration'] = data['energy_acceleration'].replace([np.inf, -np.inf], np.nan)
        new_cols.append('energy_acceleration')
        logging.info("Created 'energy_acceleration' as derivative of joint_energy over time.")
    else:
        logging.error("Missing 'continuous_frame_time' for energy_acceleration calculation.")
        sys.exit(1)
    
    # --- NEW FEATURE: Ankle Power Ratio ---
    if 'L_ANKLE_ongoing_power' in data.columns and 'R_ANKLE_ongoing_power' in data.columns:
        data['ankle_power_ratio'] = data['L_ANKLE_ongoing_power'] / (data['R_ANKLE_ongoing_power'] + 1e-6)
        new_cols.append('ankle_power_ratio')
        logging.info("Created 'ankle_power_ratio' feature comparing left to right ankle ongoing power.")
    else:
        logging.warning("Ankle ongoing power columns not found; 'ankle_power_ratio' not created.")

    # --- NEW FEATURES: Additional Asymmetry Metrics ---
    additional_asymmetry_joints = ['hip', 'ankle', 'wrist', 'elbow', 'knee', '1stfinger', '5thfinger']
    for joint in additional_asymmetry_joints:
        left_col = f"L_{joint.upper()}_energy"
        right_col = f"R_{joint.upper()}_energy"
        if left_col in data.columns and right_col in data.columns:
            col_name = f"{joint}_asymmetry"
            data[col_name] = np.abs(data[left_col] - data[right_col])
            new_cols.append(col_name)
            logging.info(f"Created asymmetry feature: {col_name}")
        else:
            logging.warning(f"Columns {left_col} and/or {right_col} not found; skipping {joint}_asymmetry.")

    # --- NEW FEATURES: Power Ratios for All Joints ---
    joints_for_power_ratio = additional_asymmetry_joints.copy()
    if 'knee' not in joints_for_power_ratio:
        joints_for_power_ratio.append('knee')
    for joint in joints_for_power_ratio:
        if joint == 'foot':
            left_col = 'left_foot_power'
            right_col = 'right_foot_power'
        else:
            left_col = f"L_{joint.upper()}_ongoing_power"
            right_col = f"R_{joint.upper()}_ongoing_power"
        logging.debug(f"Expecting power columns: {left_col} and {right_col}")
        if left_col in data.columns and right_col in data.columns:
            ratio_col = f"{joint}_power_ratio"
            data[ratio_col] = data[left_col] / (data[right_col] + 1e-6)
            new_cols.append(ratio_col)
            logging.info(f"Created power ratio feature: {ratio_col} using columns {left_col} and {right_col}")
        else:
            logging.warning(f"Columns {left_col} and/or {right_col} not found; skipping {joint}_power_ratio.")

    # --- NEW FEATURES: Side-Specific Range-of-Motion (ROM) Metrics ---
    rom_joints = {
        'KNEE': {'min': 120, 'max': 135},
        'SHOULDER': {'min': 0,  'max': 150},
        'HIP': {'min': 0,  'max': 120},
        'ANKLE': {'min': 0,  'max': 20},
        'WRIST': {'min': 0,  'max': 80}
    }
    for joint, thresholds in rom_joints.items():
        for side in ['L', 'R']:
            angle_col = f"{side}_{joint}_angle"
            if angle_col in data.columns:
                rom_col = f"{side}_{joint}_ROM"
                data[rom_col] = data.groupby('trial_id')[angle_col].transform(lambda x: x.max() - x.min())
                new_cols.append(rom_col)
                logging.info(f"Computed ROM for {side} {joint} as {rom_col}")

                deviation_col = f"{side}_{joint}_ROM_deviation"
                normal_min = thresholds['min']
                normal_max = thresholds['max']
                data[deviation_col] = np.maximum(0, normal_min - data[rom_col]) + np.maximum(0, data[rom_col] - normal_max)
                new_cols.append(deviation_col)
                logging.info(f"Computed ROM deviation for {side} {joint} as {deviation_col}")

                extreme_col = f"{side}_{joint}_ROM_extreme"
                data[extreme_col] = ((data[rom_col] < normal_min) | (data[rom_col] > normal_max)).astype(int)
                new_cols.append(extreme_col)
                logging.info(f"Created binary flag for {side} {joint} ROM extremes: {extreme_col}")
            else:
                logging.info(f"Angle column '{angle_col}' not found; skipping ROM metrics for {side} {joint}.")

    # --- Removal of Non-Contributing Features ---
    if 'wrist_angle_release' in data.columns:
        data.drop(columns=['wrist_angle_release'], inplace=True)
        logging.info("Dropped 'wrist_angle_release' column as it is not helpful for the model.")
    
    # --- Sort Data ---
    if 'continuous_frame_time' in data.columns and 'participant_id' in data.columns:
        data.sort_values(by=['participant_id', 'continuous_frame_time'], inplace=True)
        data.reset_index(drop=True, inplace=True)
        logging.info("Sorted data by 'participant_id' and 'continuous_frame_time'.")
    else:
        logging.error("Missing required columns for sorting ('participant_id', 'continuous_frame_time').")
        sys.exit(1)

    # --- Create Exhaustion Rate ---
    if 'by_trial_exhaustion_score' in data.columns and 'by_trial_time' in data.columns:
        data['exhaustion_rate'] = data['by_trial_exhaustion_score'].diff() / data['by_trial_time'].diff()
        print("print all the columns with by_trial_exhaustion_score: ", [col for col in data.columns if 'by_trial_exhaustion_score' in col])
        new_cols.append('exhaustion_rate')
        logging.info("Created 'exhaustion_rate' feature.")
    else:
        logging.error("Missing columns for 'exhaustion_rate' calculation.")
        sys.exit(1)
    
    # --- Create Simulated Heart Rate ---
    if 'by_trial_exhaustion_score' in data.columns and 'joint_energy' in data.columns:
        data['simulated_HR'] = 60 + (data['by_trial_exhaustion_score'] * 1.5) + (data['joint_energy'] * 0.3)
        new_cols.append('simulated_HR')
        logging.info("Created 'simulated_HR' feature.")
    else:
        logging.error("Missing columns for 'simulated_HR' calculation.")
        sys.exit(1)
    
    # ----- NEW: Add Grouping-Based Aggregation Features -----
    # Trial-level aggregations
    if group_trial:
        # For example, compute the trial mean exhaustion score and trial total joint energy.
        trial_aggs = data.groupby('trial_id').agg({
            'by_trial_exhaustion_score': 'mean',
            'joint_energy': 'sum'
        }).rename(columns={
            'by_trial_exhaustion_score': 'trial_mean_exhaustion',
            'joint_energy': 'trial_total_joint_energy'
        }).reset_index()
        data = data.merge(trial_aggs, on='trial_id', how='left')
        new_cols.extend(['trial_mean_exhaustion', 'trial_total_joint_energy'])
        logging.info("Added trial-level aggregated features: trial_mean_exhaustion, trial_total_joint_energy.")

    # Shot-phase-level aggregations (requires shooting_phases column)
    if group_shot_phase:
        if 'shooting_phases' in data.columns:
            shot_aggs = data.groupby(['trial_id', 'shooting_phases']).agg({
                'by_trial_exhaustion_score': 'mean',
                'joint_energy': 'sum'
            }).rename(columns={
                'by_trial_exhaustion_score': 'shot_phase_mean_exhaustion',
                'joint_energy': 'shot_phase_total_joint_energy'
            }).reset_index()
            data = data.merge(shot_aggs, on=['trial_id', 'shooting_phases'], how='left')
            new_cols.extend(['shot_phase_mean_exhaustion', 'shot_phase_total_joint_energy'])
            logging.info("Added shot-phase-level aggregated features: shot_phase_mean_exhaustion, shot_phase_total_joint_energy.")
        else:
            logging.warning("Column 'shooting_phases' not found; skipping shot-phase aggregation features.")
    

    
    _print_debug_info(step, data, new_columns=new_cols, debug=debug)
    return data



def summarize_data(data, groupby_cols, lag_columns=None, rolling_window=3, 
                   agg_columns=None, phase_list=None, global_lag=False, debug=False):
    """
    Summarize the dataset with advanced feature engineering by grouping on specified columns.
    
    This function computes:
      - The mean of selected numeric columns (except for 'injury_risk', which is aggregated using max).
      - The standard deviation of these columns (with a '_std' suffix).
      - The count of records in each group.
      - A computed duration (e.g., frame_count * 0.33 seconds if applicable).
    
    Additionally, for the columns specified in lag_columns, it computes:
      - A lag feature.
      - A delta feature (current value minus the lag).
      - A rolling average (using the specified rolling window).
    
    If a phase_list is provided and 'shooting_phases' is in groupby_cols,
    the function forces the final DataFrame to include all combinations of trial_id and
    the specified phases (Cartesian product).
    
    Parameters:
        global_lag (bool): If True, lag is computed over the entire DataFrame (across all trials);
                           if False, lag is computed within groups defined by the last groupby column.
    
    Returns:
        pd.DataFrame: Aggregated DataFrame with additional lag features.
    """
    import pandas as pd
    import numpy as np

    # 1) Set default aggregation and lag columns if not provided.
    default_agg_columns = [
        'joint_energy', 'L_ELBOW_energy', 'R_ELBOW_energy',
        'L_WRIST_energy', 'R_WRIST_energy', 'L_KNEE_energy', 'R_KNEE_energy',
        'L_HIP_energy', 'R_HIP_energy',
        'joint_power', 'L_ELBOW_ongoing_power', 'R_ELBOW_ongoing_power',
        'L_WRIST_ongoing_power', 'R_WRIST_ongoing_power',
        'L_KNEE_ongoing_power', 'R_KNEE_ongoing_power', 'L_HIP_ongoing_power', 'R_HIP_ongoing_power',
        'elbow_asymmetry', 'wrist_asymmetry', 'knee_asymmetry', 'hip_asymmetry',
        'L_ELBOW_angle', 'R_ELBOW_angle', 'L_WRIST_angle', 'R_WRIST_angle',
        'L_KNEE_angle', 'R_KNEE_angle',
        'L_SHOULDER_ROM', 'R_SHOULDER_ROM', 'L_WRIST_ROM', 'R_WRIST_ROM',
        'L_KNEE_ROM', 'R_KNEE_ROM', 'L_HIP_ROM', 'R_HIP_ROM',
        'exhaustion_rate', 'by_trial_exhaustion_score', 'injury_risk',
        'energy_acceleration', 'power_avg_5', 'rolling_power_std', 'rolling_hr_mean',
        'simulated_HR',
        'player_height_in_meters', 'player_weight__in_kg'
    ]
    default_lag_columns = [
        'joint_energy', 'joint_power',
        'L_ELBOW_energy', 'R_ELBOW_energy', 'L_WRIST_energy', 'R_WRIST_energy',
        'elbow_asymmetry', 'wrist_asymmetry',
        'L_ELBOW_angle', 'R_ELBOW_angle', 'L_WRIST_angle', 'R_WRIST_angle',
        'exhaustion_rate', 'by_trial_exhaustion_score', 'simulated_HR'
    ]
    if agg_columns is None:
        agg_columns = default_agg_columns.copy()
    if lag_columns is None:
        lag_columns = default_lag_columns.copy()

    if debug:
        print("\n--- DEBUG: summarize_data START ---")
        print(f"Initial data shape: {data.shape}")
        print(f"Grouping by: {groupby_cols}")
        print(f"Aggregation columns: {agg_columns}")
        print(f"Lag columns: {lag_columns}")
        print(f"Rolling window: {rolling_window}")
        print(f"Global lag: {global_lag}")
        if phase_list is not None:
            print(f"Forced phase list: {phase_list}")
        else:
            print("No forced phase list provided.")

    # 2) Build aggregation dictionary.
    # Use 'mean' for all columns by default, but for 'injury_risk' use 'max' to force binary outcome.
    agg_dict = {col: 'mean' for col in agg_columns}
    if 'injury_risk' in agg_columns:
        agg_dict['injury_risk'] = 'max'
    
    # 3) Group by the specified columns and compute aggregates.
    grouped = data.groupby(groupby_cols)
    mean_df = grouped.agg(agg_dict).reset_index()
    std_df = grouped[agg_columns].std(ddof=0).reset_index()
    count_df = grouped.size().reset_index(name='frame_count')

    # 4) Merge aggregates.
    summary = pd.merge(mean_df, std_df, on=groupby_cols, suffixes=("", "_std"))
    summary = pd.merge(summary, count_df, on=groupby_cols)

    # 5) Compute duration metric.
    if 'frame_count' in summary.columns:
        summary['phase_duration'] = summary['frame_count'] * 0.33

    # 6) Sort the summary by the grouping columns.
    summary = summary.sort_values(groupby_cols)

    # 7) Compute lag, delta, and rolling average features.
    for col in lag_columns:
        if col in summary.columns:
            if global_lag:
                summary[f"{col}_lag1"] = summary[col].shift(1)
                summary[f"{col}_rolling_avg"] = summary[col].rolling(window=rolling_window, min_periods=1).mean()
            else:
                group_key = groupby_cols[-1]
                summary[f"{col}_lag1"] = summary.groupby(group_key)[col].shift(1)
                summary[f"{col}_rolling_avg"] = (
                    summary.groupby(group_key)[col]
                    .rolling(window=rolling_window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
            summary[f"{col}_delta"] = summary[col] - summary[f"{col}_lag1"]
            if debug:
                print(f"\nComputed lag features for '{col}' with global_lag={global_lag}")

    # 8) Impute NaN values in lag and delta columns.
    lag_feature_cols = []
    for col in lag_columns:
        for suffix in ['_lag1', '_delta']:
            new_col = f"{col}{suffix}"
            if new_col in summary.columns:
                lag_feature_cols.append(new_col)
    for lag_col in lag_feature_cols:
        overall_mean = summary[lag_col].mean(skipna=True)
        summary[lag_col] = summary[lag_col].fillna(overall_mean)
        if debug:
            nan_count = summary[lag_col].isna().sum()
            print(f"Imputed {nan_count} NaN(s) in column '{lag_col}' with overall mean {overall_mean:.4f}")

    # 9) Handle forced phase list for shot phase data.
    if phase_list is not None and 'shooting_phases' in groupby_cols:
        trial_ids = data['trial_id'].unique()
        phase_list = np.array(phase_list)
        all_combinations = pd.MultiIndex.from_product([trial_ids, phase_list],
                                                        names=['trial_id', 'shooting_phases']).to_frame(index=False)
        summary = pd.merge(all_combinations, summary, on=['trial_id', 'shooting_phases'], how='left')
        summary = summary[summary['shooting_phases'].notnull()]

    # 10) Final sort and re-index.
    summary = summary.sort_values(groupby_cols).reset_index(drop=True)
    if debug:
        print(f"\n--- Final debug: summary at end of function ---")
        print("Final summary shape:", summary.shape)
        print("Final summary columns:", summary.columns.tolist())
        print("Sample final summary rows:\n", summary.head(10))
        print("--- DEBUG: summarize_data END ---\n")

    return summary



def add_datetime_column(data, 
                        base_datetime=pd.Timestamp('2025-01-01 00:00:00'),
                        break_seconds=10,
                        trial_id_col='trial_id',
                        freq='33ms'):
    """
    Generates a new timestamp column for the data such that within each trial the timestamps are evenly spaced
    with the specified frequency, and a fixed break (in seconds) is added between trials.
    
    Parameters:
      data (pd.DataFrame): DataFrame containing at least the trial identifier column.
      base_datetime (pd.Timestamp): The starting datetime reference.
      break_seconds (int): Number of seconds to insert as a break between trials.
      trial_id_col (str): The column in the DataFrame that identifies trials.
      freq (str): Frequency string (compatible with pd.date_range) for timestamps within each trial.
    
    Returns:
      pd.DataFrame: The original DataFrame with a new 'timestamp' column that contains evenly spaced timestamps 
                    for each trial and includes the specified break between trials.
    """
    # Get sorted unique trial identifiers. Zero-padded trial IDs sort lexicographically.
    unique_trials = sorted(data[trial_id_col].unique())
    
    # List to hold timestamp ranges for each trial.
    timestamp_list = []
    
    # Initialize the current trial start.
    current_time = base_datetime
    
    # Process each trial individually.
    for trial in unique_trials:
        # Select the rows for this trial.
        trial_mask = data[trial_id_col] == trial
        trial_data = data[trial_mask]
        n = len(trial_data)
        
        # Generate a date_range for this trial with n periods at the specified frequency.
        trial_timestamps = pd.date_range(start=current_time, periods=n, freq=freq)
        timestamp_list.append(trial_timestamps)
        
        # Set the start time for the next trial:
        # current_time becomes the last timestamp of this trial plus the fixed break.
        current_time = trial_timestamps[-1] + pd.Timedelta(seconds=break_seconds) + pd.Timedelta(milliseconds=33)
    
    # Construct a single timestamp Series aligned with the original data.
    timestamps = pd.Series(index=data.index, dtype='datetime64[ns]')
    for trial, trial_timestamps in zip(unique_trials, timestamp_list):
        indices = data.index[data[trial_id_col] == trial]
        timestamps.loc[indices] = trial_timestamps
        

    data = data.copy()
    data['timestamp'] = timestamps
    # Set the 'datetime' column as index, then force a uniform frequency.
    # After setting the index and applying asfreq:
    # data = data.set_index('timestamp')
    return data



def feature_engineering(data, window_size=5, debug=False, group_trial=False, group_shot_phase=False):
    """Optimized feature engineering with vectorized operations.
    
    NEW GROUPING FEATURE (optional):
      - If group_trial is True, adds trial-level aggregated features (mean exhaustion and injury rate).
      - If group_shot_phase is True, and if 'shooting_phases' exists, adds shot-phase-level aggregated features.
    
    Parameters:
      - data (pd.DataFrame): Input DataFrame.
      - window_size (int): Window size for some rolling computations.
      - debug (bool): If True, prints detailed debug outputs.
      - group_trial (bool): If True, add trial-level aggregated features.
      - group_shot_phase (bool): If True, add shot-phase-level aggregated features.
    
    Returns:
      - data (pd.DataFrame): Updated DataFrame with engineered features and optional grouping-based columns.
    """
    step = "feature_engineering"
    new_cols = []
    rolling_window = 20
    required_columns = {
        'base': ['by_trial_exhaustion_score', 'joint_power', 'simulated_HR', 'continuous_frame_time'],
        'joints': ['by_trial_time']
    }
    
    # Validate required columns
    missing = [col for col in required_columns['base'] if col not in data.columns]
    if missing:
        logging.error(f"Missing required columns: {missing}")
        sys.exit(1)

    # Vectorized temporal features
    data['time_since_start'] = data['continuous_frame_time'] - data['continuous_frame_time'].min()
    new_cols.append('time_since_start')
    
    # Fill ball-related columns with 0 when not in play
    ball_cols = ['ball_speed', 'ball_velocity_x', 'ball_velocity_y', 'ball_velocity_z']
    data[ball_cols] = data[ball_cols].fillna(0)

    # For motion columns, forward-fill missing values
    motion_cols = ['dx', 'dy', 'dz']
    data[motion_cols] = data[motion_cols].ffill().fillna(0)


    # Rolling features
    roll_config = {
        'power_avg_5': ('joint_power', 'mean'),
        'rolling_power_std': ('joint_power', 'std'),
        'rolling_hr_mean': ('simulated_HR', 'mean')
    }
    for new_col, (base_col, func) in roll_config.items():
        data[new_col] = getattr(data[base_col].rolling(window_size, min_periods=1), func)()
    
    # Safe expanding quantile function
    def safe_expanding_quantile(s):
        return s.expanding().quantile(0.75).shift().fillna(0)
    
    # Optional new feature: Rolling Energy Standard Deviation
    if 'joint_energy' in data.columns:
        data['rolling_energy_std'] = data['joint_energy'].rolling(window=window_size, min_periods=1).std(ddof=0)
        logging.info(f"Created 'rolling_energy_std' with sample: {data['rolling_energy_std'].head(10).tolist()}")
        logging.info(f"Created 'rolling_energy_std' with window {window_size}.")
    else:
        logging.warning("Column 'joint_energy' missing for 'rolling_energy_std'.")
    new_cols.append('rolling_energy_std')
    
    # Vectorized exhaustion features
    data['exhaustion_lag1'] = data['by_trial_exhaustion_score'].shift(1)
    data['ema_exhaustion'] = data['by_trial_exhaustion_score'].ewm(span=10, adjust=False).mean()
    data['rolling_exhaustion'] = data['by_trial_exhaustion_score'].rolling(rolling_window, min_periods=1).sum()
    
    # Injury risk calculation
    data['injury_risk'] = (data['rolling_exhaustion'] > safe_expanding_quantile(data['rolling_exhaustion'])).astype(int)
    new_cols += ['exhaustion_lag1', 'ema_exhaustion', 'rolling_exhaustion', 'injury_risk']

    # Joint features computed per joint & side
    joints = ['ANKLE', 'WRIST', 'ELBOW', 'KNEE', 'HIP']
    sides = ['L', 'R']
    dt = data['by_trial_time'].diff().replace(0, np.nan)
    for joint in joints:
        for side in sides:
            joint_name = f"{side}_{joint}"
            score_col = f'{joint_name}_energy_by_trial_exhaustion_score'
            if score_col not in data.columns:
                continue
            data[f'{joint_name}_exhaustion_rate'] = data[score_col].diff() / dt
            data[f'{joint_name}_rolling_exhaustion'] = data[score_col].rolling(rolling_window, min_periods=1).sum()
            rolling_series = data[f'{joint_name}_rolling_exhaustion']
            data[f'{joint_name}_injury_risk'] = (rolling_series > safe_expanding_quantile(rolling_series)).astype(int)
            new_cols.extend([f'{joint_name}_exhaustion_rate', f'{joint_name}_rolling_exhaustion', f'{joint_name}_injury_risk'])

    # Drop rows with NA in exhaustion_lag1
    data.dropna(subset=['exhaustion_lag1'], inplace=True)
    
    # ----- NEW: Add Grouping-Based Aggregation Features in Feature Engineering -----
    # Trial-level aggregations for feature engineering
    if group_trial:
        trial_aggs = data.groupby('trial_id').agg({
            'by_trial_exhaustion_score': 'mean',
            'injury_risk': 'mean'
        }).rename(columns={
            'by_trial_exhaustion_score': 'trial_mean_exhaustion_fe',
            'injury_risk': 'trial_injury_rate_fe'
        }).reset_index()
        data = data.merge(trial_aggs, on='trial_id', how='left')
        new_cols.extend(['trial_mean_exhaustion_fe', 'trial_injury_rate_fe'])
        logging.info("Added trial-level aggregated features in feature_engineering: trial_mean_exhaustion_fe, trial_injury_rate_fe.")

    # Shot-phase-level aggregations
    if group_shot_phase:
        if 'shooting_phases' in data.columns:
            shot_aggs = data.groupby(['trial_id', 'shooting_phases']).agg({
                'by_trial_exhaustion_score': 'mean',
                'injury_risk': 'mean'
            }).rename(columns={
                'by_trial_exhaustion_score': 'shot_phase_mean_exhaustion_fe',
                'injury_risk': 'shot_phase_injury_rate_fe'
            }).reset_index()
            data = data.merge(shot_aggs, on=['trial_id', 'shooting_phases'], how='left')
            new_cols.extend(['shot_phase_mean_exhaustion_fe', 'shot_phase_injury_rate_fe'])
            logging.info("Added shot-phase-level aggregated features in feature_engineering: shot_phase_mean_exhaustion_fe, shot_phase_injury_rate_fe.")
        else:
            logging.warning("Column 'shooting_phases' not found; skipping shot-phase grouping features in feature_engineering.")
    
    # Remove duplicate columns, if any.
    data = data.loc[:, ~data.columns.duplicated()]

    ## Add datetime column using our new function
    data = add_datetime_column(data, base_datetime=pd.Timestamp('2025-01-01 00:00:00'), 
                               break_seconds=0, trial_id_col='trial_id', freq='33ms')
    print("Data columns for Darts processing:", data.columns.tolist())
        
    if debug:
        _print_debug_info(step, data, new_columns=new_cols, debug=debug)
    
    return data


def make_exhaustion_monotonic_and_time_to_zero(data):
    # (A) Cumulative exhaustion example
    data['cumulative_exhaustion'] = (
        data.groupby('participant_id')['by_trial_exhaustion_score']
            .cumsum()
    )
    
    # (B) Invert the raw exhaustion so that 1=Fresh, 0=Exhausted
    data['remaining_capacity'] = 1.0 - data['by_trial_exhaustion_score']
    
    # (C) Compute "time to 0 exhaustion"
    data = data.sort_values(['participant_id', 'continuous_frame_time']).reset_index(drop=True)
    times = data['continuous_frame_time'].values
    exhaustion = data['by_trial_exhaustion_score'].values
    time_to_zero = np.full(len(data), np.nan)
    
    for i in range(len(data)):
        if exhaustion[i] <= 0.0:
            time_to_zero[i] = 0.0
        else:
            future_idxs = np.where(exhaustion[i:] <= 0.0)[0]
            if len(future_idxs) > 0:
                j = i + future_idxs[0]
                time_to_zero[i] = times[j] - times[i]
            else:
                # If it never reaches 0 in the future, leave it as NaN or set a default
                time_to_zero[i] = np.nan

    data['time_to_zero_exhaustion'] = time_to_zero
    
    return data



def add_simulated_player_metrics(df, window=5, debug=False):
    """
    Adds simulated player metrics to mimic heart rate and fatigue.
    
    New Metrics:
      - simulated_HR_fake: Alternative simulated heart rate.
      - fatigue_index_fake: Combined fatigue index.
      - fatigue_rate_fake: Frame-by-frame rate of change of fatigue_index_fake.
      - HR_variability_fake: Rolling standard deviation of simulated_HR_fake.
    
    Parameters:
      - df (pd.DataFrame): DataFrame with required columns (e.g., by_trial_exhaustion_score, joint_energy, overall_exhaustion_score, dt).
      - window (int): Rolling window size for HR variability.
      - debug (bool): If True, prints detailed debug outputs.
    
    Returns:
      - df (pd.DataFrame): DataFrame with new simulated metrics.
    """
    step = "add_simulated_player_metrics"
    new_cols = []
    
    # Use maximum joint_energy for scaling
    max_joint_energy = df['joint_energy'].max() if 'joint_energy' in df.columns else 1
    df['simulated_HR_fake'] = 60 + (df['by_trial_exhaustion_score'] * 2.0) + ((df['joint_energy'] / max_joint_energy) * 20)
    new_cols.append('simulated_HR_fake')
    
    df['fatigue_index_fake'] = df['overall_exhaustion_score'] + ((df['simulated_HR_fake'] - 60) / 100)
    new_cols.append('fatigue_index_fake')
    
    df['fatigue_rate_fake'] = df['fatigue_index_fake'].diff() / df['dt']
    df['fatigue_rate_fake'] = df['fatigue_rate_fake'].fillna(0)
    new_cols.append('fatigue_rate_fake')
    
    df['HR_variability_fake'] = df['simulated_HR_fake'].rolling(window=window, min_periods=1).std()
    new_cols.append('HR_variability_fake')
    
    _print_debug_info(step, df, new_columns=new_cols, debug=debug)
    return df


def check_and_drop_nulls(df, columns_to_drop=None, df_name="DataFrame", debug=False):
    """
    Prints a summary of null values for all columns in the DataFrame.
    Optionally drops rows with nulls in the specified columns and then prints
    the updated null summary.

    Parameters:
      df (pd.DataFrame): The DataFrame to check.
      columns_to_drop (list, optional): List of columns in which, if nulls are present,
                                        the corresponding rows should be dropped.
                                        If None, no rows will be dropped.
      df_name (str, optional): Name for the DataFrame used in print messages.

    Returns:
      pd.DataFrame: The resulting DataFrame after dropping rows (if columns_to_drop is provided).
    """
    total_rows = len(df)
    print(f"\nNull summary for {df_name}: Total Rows = {total_rows}")
    
    # Print summary for all columns.
    null_sums = df.isnull().sum()
    for col, count in null_sums.items():
        percent = (count / total_rows) * 100
        if debug:
            print(f"{col}: {count} nulls, {percent:.2f}% null")
    
    if columns_to_drop:
        df = df.dropna(subset=columns_to_drop)
        print(f"\nAfter dropping rows with nulls in columns: {columns_to_drop}")
        total_rows = len(df)
        print(f"Total Rows = {total_rows}")
        null_sums = df.isnull().sum()
        for col, count in null_sums.items():
            percent = (count / total_rows) * 100
            print(f"{col}: {count} nulls, {percent:.2f}% null")
    
    return df


def clip_outliers(df, column, k=3.0, min_value=0.0):
    """
    Clips outliers in a given column to the IQR-based range [max(Q1 - k*IQR, min_value), Q3 + k*IQR].
    Ensures the lower bound is at least `min_value`, preventing negative values if min_value=0.0.
    
    Parameters:
      - df (pd.DataFrame): The DataFrame with data.
      - column (str): Name of the column to clip.
      - k (float): Multiplier for the IQR to define the clip boundaries.
      - min_value (float): Minimum allowed value (defaults to 0.0).
    
    Returns:
      - df (pd.DataFrame): Modified DataFrame with clipped column values.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    # Compute the IQR-based bounds
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    # Enforce a hard floor at min_value (e.g., 0.0)
    lower_bound = max(lower_bound, min_value)
    
    df[column] = df[column].clip(lower_bound, upper_bound)
    return df



def prepare_base_datasets(csv_path, json_path, debug=False):
    # 1) Load and base‐engineer:
    data = load_data(csv_path, json_path, debug=debug)
    data = prepare_joint_features(data, debug=debug)
    data = feature_engineering(data, debug=debug)
    data = check_and_drop_nulls(data,
                                columns_to_drop=['energy_acceleration', 'exhaustion_rate'],
                                df_name="Final Data")

    # 2) Trial‐summary:
    trial = prepare_joint_features(data, debug=debug, group_trial=True)
    trial = feature_engineering(trial, debug=debug, group_trial=True)
    trial_summary = summarize_data(
        data,
        groupby_cols=['trial_id'],
        lag_columns=[
        'joint_energy', 'L_ELBOW_energy', 'R_ELBOW_energy', 'L_WRIST_energy', 'R_WRIST_energy',
        'L_KNEE_energy', 'R_KNEE_energy', 'L_HIP_energy', 'R_HIP_energy',
        'joint_power', 'L_ELBOW_ongoing_power', 'R_ELBOW_ongoing_power', 
        'L_WRIST_ongoing_power', 'R_WRIST_ongoing_power',
        'L_KNEE_ongoing_power', 'R_KNEE_ongoing_power', 'L_HIP_ongoing_power', 'R_HIP_ongoing_power',
        'elbow_asymmetry', 'wrist_asymmetry', 'knee_asymmetry', 'hip_asymmetry',
        'L_ELBOW_angle', 'R_ELBOW_angle', 'L_WRIST_angle', 'R_WRIST_angle', 
        'L_KNEE_angle', 'R_KNEE_angle',
        'L_SHOULDER_ROM', 'R_SHOULDER_ROM', 'L_WRIST_ROM', 'R_WRIST_ROM',
        'L_KNEE_ROM', 'R_KNEE_ROM', 'L_HIP_ROM', 'R_HIP_ROM',
        'exhaustion_rate', 'by_trial_exhaustion_score', 'injury_risk',
        'energy_acceleration', 'power_avg_5', 'rolling_power_std', 'rolling_hr_mean',
        'simulated_HR',
        'player_height_in_meters', 'player_weight__in_kg'
    ],
        rolling_window=3,
        agg_columns= [
        'joint_energy', 'L_ELBOW_energy', 'R_ELBOW_energy', 'L_WRIST_energy', 'R_WRIST_energy',
        'L_KNEE_energy', 'R_KNEE_energy', 'L_HIP_energy', 'R_HIP_energy',
        'joint_power', 'L_ELBOW_ongoing_power', 'R_ELBOW_ongoing_power', 
        'L_WRIST_ongoing_power', 'R_WRIST_ongoing_power',
        'L_KNEE_ongoing_power', 'R_KNEE_ongoing_power', 'L_HIP_ongoing_power', 'R_HIP_ongoing_power',
        'elbow_asymmetry', 'wrist_asymmetry', 'knee_asymmetry', 'hip_asymmetry',
        'L_ELBOW_angle', 'R_ELBOW_angle', 'L_WRIST_angle', 'R_WRIST_angle', 
        'L_KNEE_angle', 'R_KNEE_angle',
        'L_SHOULDER_ROM', 'R_SHOULDER_ROM', 'L_WRIST_ROM', 'R_WRIST_ROM',
        'L_KNEE_ROM', 'R_KNEE_ROM', 'L_HIP_ROM', 'R_HIP_ROM',
        'exhaustion_rate', 'by_trial_exhaustion_score', 'injury_risk',
        'energy_acceleration', 'power_avg_5', 'rolling_power_std', 'rolling_hr_mean',
        'simulated_HR',
        'player_height_in_meters', 'player_weight__in_kg'
    ],
        global_lag=True,
        debug=debug
    )

    # 3) Shot‐phase‐summary:
    shot = prepare_joint_features(data, debug=debug, group_trial=True, group_shot_phase=True)
    shot = feature_engineering(shot, debug=debug, group_trial=True, group_shot_phase=True)
    shot_summary = summarize_data(
        shot,
        groupby_cols=['trial_id', 'shooting_phases'],
        lag_columns=[
        'joint_energy', 'L_ELBOW_energy', 'R_ELBOW_energy', 'L_WRIST_energy', 'R_WRIST_energy',
        'L_KNEE_energy', 'R_KNEE_energy', 'L_HIP_energy', 'R_HIP_energy',
        'joint_power', 'L_ELBOW_ongoing_power', 'R_ELBOW_ongoing_power', 
        'L_WRIST_ongoing_power', 'R_WRIST_ongoing_power',
        'L_KNEE_ongoing_power', 'R_KNEE_ongoing_power', 'L_HIP_ongoing_power', 'R_HIP_ongoing_power',
        'elbow_asymmetry', 'wrist_asymmetry', 'knee_asymmetry', 'hip_asymmetry',
        'L_ELBOW_angle', 'R_ELBOW_angle', 'L_WRIST_angle', 'R_WRIST_angle', 
        'L_KNEE_angle', 'R_KNEE_angle',
        'L_SHOULDER_ROM', 'R_SHOULDER_ROM', 'L_WRIST_ROM', 'R_WRIST_ROM',
        'L_KNEE_ROM', 'R_KNEE_ROM', 'L_HIP_ROM', 'R_HIP_ROM',
        'exhaustion_rate', 'by_trial_exhaustion_score', 'injury_risk',
        'energy_acceleration', 'power_avg_5', 'rolling_power_std', 'rolling_hr_mean',
        'simulated_HR',
        'player_height_in_meters', 'player_weight__in_kg'
    ],
        rolling_window=3,
        agg_columns= [
        'joint_energy', 'L_ELBOW_energy', 'R_ELBOW_energy', 'L_WRIST_energy', 'R_WRIST_energy',
        'L_KNEE_energy', 'R_KNEE_energy', 'L_HIP_energy', 'R_HIP_energy',
        'joint_power', 'L_ELBOW_ongoing_power', 'R_ELBOW_ongoing_power', 
        'L_WRIST_ongoing_power', 'R_WRIST_ongoing_power',
        'L_KNEE_ongoing_power', 'R_KNEE_ongoing_power', 'L_HIP_ongoing_power', 'R_HIP_ongoing_power',
        'elbow_asymmetry', 'wrist_asymmetry', 'knee_asymmetry', 'hip_asymmetry',
        'L_ELBOW_angle', 'R_ELBOW_angle', 'L_WRIST_angle', 'R_WRIST_angle', 
        'L_KNEE_angle', 'R_KNEE_angle',
        'L_SHOULDER_ROM', 'R_SHOULDER_ROM', 'L_WRIST_ROM', 'R_WRIST_ROM',
        'L_KNEE_ROM', 'R_KNEE_ROM', 'L_HIP_ROM', 'R_HIP_ROM',
        'exhaustion_rate', 'by_trial_exhaustion_score', 'injury_risk',
        'energy_acceleration', 'power_avg_5', 'rolling_power_std', 'rolling_hr_mean',
        'simulated_HR',
        'player_height_in_meters', 'player_weight__in_kg'
    ],
        phase_list=["arm_cock", "arm_release", "leg_cock", "wrist_release"],
        debug=debug
    )
    data = check_and_drop_nulls(data, columns_to_drop=['energy_acceleration', 'exhaustion_rate'], df_name="Final Data")
    # Example usage
    joints = ['ANKLE', 'WRIST', 'ELBOW', 'KNEE', 'HIP']
    sides = ['L', 'R']
    data = clip_outliers(data, 'exhaustion_rate', k=3.0, min_value=0.0)
    data = clip_outliers(data, 'joint_power', k=3.0, min_value=0.0)
    data = clip_outliers(data, 'joint_energy', k=3.0, min_value=0.0)
    for joint in joints:
        for side in sides:
            data = clip_outliers(data, f'{side}_{joint}_exhaustion_rate', k=3.0, min_value=0.0)
            
    return data, trial_summary, shot_summary



def joint_specific_analysis(data, joint_energy_columns, debug=False):
    """
    Performs joint-specific analysis including:
      - Energy distribution per joint.
      - Injury risk analysis for each joint.
      - Cumulative energy accumulation patterns.
    
    Parameters:
      - data (pd.DataFrame): Input DataFrame.
      - joint_energy_columns (list): List of joint energy column names.
      - debug (bool): If True, prints debug information.
    """
    step = "joint_specific_analysis"
    # Energy distribution across joints
    joint_energy_melted = data[joint_energy_columns].melt(var_name='Joint', value_name='Energy')
    plt.figure(figsize=(15, 8))
    order = joint_energy_melted.groupby('Joint')['Energy'].median().sort_values().index
    sns.boxplot(x='Joint', y='Energy', data=joint_energy_melted, order=order)
    plt.title('Joint Energy Distributions (Sorted by Median Energy)')
    plt.xticks(rotation=45)
    plt.show()
    if debug:
        logging.info("Displayed boxplot for joint energy distributions.")
    else:
        logging.info("Energy distribution plot displayed.")
    
    # Injury risk analysis: only run if 'injury_risk' exists.
    if 'injury_risk' not in data.columns:
        logging.warning("Column 'injury_risk' not found; skipping injury risk analysis in joint_specific_analysis.")
    else:
        num_plots = len(joint_energy_columns)
        ncols = 4
        nrows = int(np.ceil(num_plots / ncols))
        plt.figure(figsize=(15, 10))
        for i, joint in enumerate(joint_energy_columns, 1):
            plt.subplot(nrows, ncols, i)
            sns.boxplot(x='injury_risk', y=joint, data=data)
            plt.title(f'{joint.split("_")[0].title()} Energy')
            plt.tight_layout()
        plt.suptitle('Joint Energy Distributions by Injury Risk', y=1.02)
        plt.show()
        if debug:
            logging.info("Displayed injury risk analysis plots for joint energy.")
        else:
            logging.info("Injury risk analysis plots displayed.")
    
    # Cumulative energy accumulation patterns
    joint_cumulative = data.groupby('participant_id')[joint_energy_columns].cumsum()
    joint_cumulative['time'] = data['continuous_frame_time']
    joint_cumulative_melted = joint_cumulative.melt(id_vars='time', var_name='Joint', value_name='Cumulative Energy')
    plt.figure(figsize=(15, 8))
    sns.lineplot(x='time', y='Cumulative Energy', hue='Joint', 
                 data=joint_cumulative_melted, estimator='median', errorbar=None)
    plt.title('Cumulative Joint Energy Over Time (Median Across Participants)')
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Energy')
    plt.show()
    if debug:
        logging.info("Displayed cumulative joint energy plot.")
    else:
        logging.info("Cumulative energy plot displayed.")
    
    _print_debug_info(step, data, debug=debug)


def movement_pattern_analysis(data, debug=False):
    """
    Performs movement pattern analysis:
      - Angular velocity histograms with KDE.
      - Asymmetry analysis via pairplot.
    
    Parameters:
      - data (pd.DataFrame): Input DataFrame.
      - debug (bool): If True, prints debug information.
    """
    step = "movement_pattern_analysis"
    # Angular velocity analysis
    angular_columns = [col for col in data.columns if '_angular_velocity' in col]
    if angular_columns:
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 15))
        axes = axes.flatten()
        for ax, col in zip(axes, angular_columns):
            sns.histplot(data[col], ax=ax, kde=True)
            ax.set_title(f'{col.split("_")[0].title()} Angular Velocity')
        for j in range(len(angular_columns), len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()
        logging.info("Displayed angular velocity histograms.")
    else:
        logging.info("No angular velocity columns found.")
    
    # Asymmetry analysis
    asymmetry_metrics = [col for col in data.columns if 'asymmetry' in col]
    if 'injury_risk' in data.columns and asymmetry_metrics:
        sns.pairplot(data[asymmetry_metrics + ['injury_risk']], hue='injury_risk', corner=True)
        plt.suptitle('Joint Asymmetry Relationships with Injury Risk', y=1.02)
        plt.show()
        logging.info("Displayed asymmetry pairplot.")
    else:
        logging.info("Required columns for asymmetry analysis not found.")
    
    _print_debug_info(step, data, debug=debug)


def temporal_analysis_enhancements(data, debug=False):
    """
    Performs temporal analysis enhancements:
      - Computes lagged correlations between joint energy and exhaustion score.
      - Plots autocorrelation of joint energy.
    
    Parameters:
      - data (pd.DataFrame): Input DataFrame.
      - debug (bool): If True, prints debug information.
    """
    step = "temporal_analysis_enhancements"
    max_lag = 10
    lagged_corrs = []
    for lag in range(1, max_lag + 1):
        corr_val = data['joint_energy'].corr(data['by_trial_exhaustion_score'].shift(lag))
        lagged_corrs.append(corr_val)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_lag + 1), lagged_corrs, marker='o')
    plt.title('Lagged Correlation Between Joint Energy and Exhaustion Score')
    plt.xlabel('Time Lag (periods)')
    plt.ylabel('Correlation Coefficient')
    plt.grid(True)
    plt.show()
    
    from statsmodels.graphics.tsaplots import plot_acf
    plt.figure(figsize=(12, 6))
    plot_acf(data['joint_energy'].dropna(), lags=50, alpha=0.05)
    plt.title('Joint Energy Autocorrelation')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.show()
    
    _print_debug_info(step, data, debug=debug)


def multivariate_analysis(data, joint_energy_columns, debug=False):
    """
    Performs multivariate analysis separately for left- and right-sided joints:
      - 3D visualization of joint energy interactions for each side.
      - KMeans clustering on selected features for each side.
    
    Parameters:
      - data (pd.DataFrame): Input DataFrame.
      - joint_energy_columns (list): List of all joint energy columns.
      - debug (bool): If True, prints debug information.
    """
    step = "multivariate_analysis"

    # --- 3D Visualization: Left Side ---
    required_left = ['L_ELBOW_energy', 'L_KNEE_energy', 'L_ANKLE_energy', 'injury_risk']
    if all(col in data.columns for col in required_left):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data['L_ELBOW_energy'], 
                             data['L_KNEE_energy'], 
                             data['L_ANKLE_energy'], 
                             c=data['injury_risk'],
                             cmap='viridis',
                             alpha=0.6)
        ax.set_xlabel('L Elbow Energy')
        ax.set_ylabel('L Knee Energy')
        ax.set_zlabel('L Ankle Energy')
        plt.title('3D Joint Energy Space (Left Side) with Injury Risk Coloring')
        plt.colorbar(scatter, label='Injury Risk')
        plt.show()
    else:
        logging.info("Required left-side columns for 3D analysis not found; skipping left side 3D plot.")

    # --- 3D Visualization: Right Side ---
    required_right = ['R_ELBOW_energy', 'R_KNEE_energy', 'R_ANKLE_energy', 'injury_risk']
    if all(col in data.columns for col in required_right):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data['R_ELBOW_energy'], 
                             data['R_KNEE_energy'], 
                             data['R_ANKLE_energy'], 
                             c=data['injury_risk'],
                             cmap='viridis',
                             alpha=0.6)
        ax.set_xlabel('R Elbow Energy')
        ax.set_ylabel('R Knee Energy')
        ax.set_zlabel('R Ankle Energy')
        plt.title('3D Joint Energy Space (Right Side) with Injury Risk Coloring')
        plt.colorbar(scatter, label='Injury Risk')
        plt.show()
    else:
        logging.info("Required right-side columns for 3D analysis not found; skipping right side 3D plot.")

    # --- Clustering Analysis: Left Side ---
    left_features = ['L_ELBOW_energy', 'L_KNEE_energy', 'L_ANKLE_energy']
    # Optionally include asymmetry features if desired (they compare L vs R)
    left_features = [feat for feat in left_features if feat in data.columns]
    if left_features:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        X_left = data[left_features].dropna()
        X_left_scaled = StandardScaler().fit_transform(X_left)
        kmeans_left = KMeans(n_clusters=3, random_state=42).fit(X_left_scaled)
        data.loc[X_left.index, 'left_movement_cluster'] = kmeans_left.labels_
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='L_ELBOW_energy', y='L_KNEE_energy', hue='left_movement_cluster', 
                        data=data, palette='viridis', alpha=0.6)
        plt.title('Left Side Movement Clusters in Elbow-Knee Energy Space')
        plt.xlabel('L Elbow Energy')
        plt.ylabel('L Knee Energy')
        plt.show()
    else:
        logging.info("Not enough left-side features available for clustering analysis.")

    # --- Clustering Analysis: Right Side ---
    right_features = ['R_ELBOW_energy', 'R_KNEE_energy', 'R_ANKLE_energy']
    right_features = [feat for feat in right_features if feat in data.columns]
    if right_features:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        X_right = data[right_features].dropna()
        X_right_scaled = StandardScaler().fit_transform(X_right)
        kmeans_right = KMeans(n_clusters=3, random_state=42).fit(X_right_scaled)
        data.loc[X_right.index, 'right_movement_cluster'] = kmeans_right.labels_
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='R_ELBOW_energy', y='R_KNEE_energy', hue='right_movement_cluster', 
                        data=data, palette='viridis', alpha=0.6)
        plt.title('Right Side Movement Clusters in Elbow-Knee Energy Space')
        plt.xlabel('R Elbow Energy')
        plt.ylabel('R Knee Energy')
        plt.show()
    else:
        logging.info("Not enough right-side features available for clustering analysis.")

    _print_debug_info(step, data, debug=debug)



def statistical_testing(data, joint_energy_columns, debug=False):
    """
    Performs Mann-Whitney U tests on each joint energy metric between low and high injury risk groups.
    
    Parameters:
      - data (pd.DataFrame): Input DataFrame.
      - joint_energy_columns (list): List of joint energy column names.
      - debug (bool): If True, prints detailed test outputs.
    
    Returns:
      - results_df (pd.DataFrame): Summary table of test statistics.
    """
    from scipy.stats import mannwhitneyu
    step = "statistical_testing"
    results = []
    for joint in joint_energy_columns:
        if joint in data.columns and 'injury_risk' in data.columns:
            low_risk = data[data['injury_risk'] == 0][joint]
            high_risk = data[data['injury_risk'] == 1][joint]
            stat, p = mannwhitneyu(low_risk, high_risk, alternative='two-sided')
            effect_size = stat / (len(low_risk) * len(high_risk)) if (len(low_risk) * len(high_risk)) > 0 else np.nan
            results.append({
                'Joint': joint.split('_')[0],
                'U Statistic': stat,
                'p-value': p,
                'Effect Size': effect_size
            })
    results_df = pd.DataFrame(results).sort_values('p-value')
    logging.info("Mann-Whitney U Test Results:")
    logging.info(results_df)
    _print_debug_info(step, data, debug=debug)
    return results_df



## Utility function to print debug information
def validate_target_column(target_array, target_name, expected_type):
    """
    Validates that the target array meets the expected type.
    
    Parameters:
      - target_array (np.ndarray): The target values.
      - target_name (str): Name of the target (for logging).
      - expected_type (str): Either "binary" or "continuous".
    
    Raises:
      - ValueError: If the target values do not match the expectation.
    """
    unique_vals = np.unique(target_array)
    logging.info(f"Validating target '{target_name}': unique values = {unique_vals}")
    
    if expected_type == "binary":
        # Check that the unique values are a subset of {0, 1}
        if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"Target '{target_name}' expected to be binary, but got values: {unique_vals}")
    elif expected_type == "continuous":
        # For continuous targets, if there are only two unique values (0 and 1), warn or error.
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"Target '{target_name}' expected to be continuous, but appears binary: {unique_vals}")
    else:
        raise ValueError(f"Unknown expected_type '{expected_type}' for target '{target_name}'")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, probplot

def check_distribution_and_zscore(df, col_list, zscore_suffix='_zscore'):
    """
    For each column in col_list, this function:
      - Prints summary stats
      - Plots histogram + KDE
      - Performs Shapiro-Wilk test for normality
      - Plots QQ-plot
      - Creates a z-score column in the DataFrame
    """
    results = {}
    for col in col_list:
        print(f"\n--- Analyzing Column: {col} ---")

        # 1. Summary statistics
        desc = df[col].describe()
        print(desc)

        # 2. Plot histogram and KDE
        plt.figure(figsize=(7,4))
        df[col].dropna().hist(bins=30, alpha=0.5, density=True, label=f"{col} Histogram")
        df[col].dropna().plot(kind='kde', label=f"{col} KDE")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.show()

        # 3. Normality check (Shapiro-Wilk)
        #    Note: Shapiro test can be sensitive to large n. If you have a lot of rows,
        #    you might want to sample or use another normality test.
        sample_for_shapiro = df[col].dropna()
        if len(sample_for_shapiro) > 3:
            stat, p_value = shapiro(sample_for_shapiro)
            print(f"Shapiro-Wilk p-value for {col}: {p_value:.4f}")
        else:
            stat, p_value = np.nan, np.nan
            print(f"Not enough non-null values for Shapiro test in {col}.")

        # 4. QQ-plot (to visually check normality)
        plt.figure(figsize=(5,5))
        probplot(sample_for_shapiro, dist="norm", plot=plt)
        plt.title(f"QQ-Plot of {col}")
        plt.grid(True)
        plt.show()

        # 5. Compute z-score and attach to DataFrame
        mean_val = df[col].mean()
        std_val = df[col].std(ddof=0)  # population-like; use ddof=1 for sample
        z_column = col + zscore_suffix
        df[z_column] = (df[col] - mean_val) / (std_val if std_val != 0 else 1e-8)

        # identify potential outliers: e.g., absolute z-score > 3
        outliers = df[z_column].abs() > 3
        outlier_count = outliers.sum()
        outlier_pct = 100.0 * outlier_count / len(df)
        print(f"Number of potential outliers for {col} (|z|>3): {outlier_count} ({outlier_pct:.2f}%)")

        # Store stats in a results dictionary if you want to use them later
        results[col] = {
            'mean': mean_val,
            'std': std_val,
            'shapiro_stat': stat,
            'shapiro_p': p_value,
            'outlier_count': outlier_count,
            'outlier_pct': outlier_pct
        }
    return df, pd.DataFrame(results).T


###############################################################################
# NEW FUNCTION FOR SCALING DATA BEFORE LSTM
###############################################################################
def scale_for_lstm(df, columns_to_scale, method="standard", debug=False):
    """
    Scales specified columns for LSTM training, either by z-score (StandardScaler)
    or min-max normalization (MinMaxScaler).
    
    Parameters:
      - df (pd.DataFrame): The DataFrame containing the features to be scaled.
      - columns_to_scale (list of str): Columns to scale.
      - method (str): "standard" for StandardScaler (z-score), "minmax" for MinMaxScaler.
      - debug (bool): If True, prints debug information about scaling.
      
    Returns:
      - scaled_df (pd.DataFrame): Copy of the original DataFrame with scaled columns.
      - scaler (object): The fitted scaler object (useful if you need inverse transforms later).
    
    Note:
      This function does NOT drop or rename columns; it overwrites them with scaled values
      to keep them numeric but on a uniform scale.
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    # 1) Make a copy so we don't mutate the original
    scaled_df = df.copy()
    
    # 2) Choose which scaler to use
    if method == "standard":
        scaler = StandardScaler()
        if debug:
            print("Using StandardScaler for columns:", columns_to_scale)
    elif method == "minmax":
        scaler = MinMaxScaler()
        if debug:
            print("Using MinMaxScaler for columns:", columns_to_scale)
    else:
        raise ValueError(f"Unknown scaling method '{method}'. Choose 'standard' or 'minmax'.")
    
    # 3) Fit and transform the specified columns
    # Drop NaNs to avoid issues with partial columns, then reassign
    subset = scaled_df[columns_to_scale].dropna()
    scaled_values = scaler.fit_transform(subset)
    
    # Put scaled values back into the DataFrame (at the same indices)
    scaled_df.loc[subset.index, columns_to_scale] = scaled_values
    
    if debug:
        print(f"Scaler mean_ (if standard) or data_min_ (if minmax): {scaler.mean_ if method=='standard' else scaler.data_min_}")
        print(f"Scaler scale_ (if standard) or data_range_ (if minmax): {scaler.scale_ if method=='standard' else scaler.data_range_}")
        print("Sample after scaling:\n", scaled_df[columns_to_scale].head(5))
    
    return scaled_df, scaler






###############################################################################
# MAIN SCRIPT
###############################################################################

if __name__ == "__main__":
    # Run the main pipeline with debug output enabled.

    debug=True
 
    # """
    # Main processing pipeline:
    #   1. Loads and merges data.
    #   2. Prepares joint features.
    #   3. Performs feature engineering.
    #   4. Adds simulated player metrics.
    #   5. Executes various analyses (joint-specific, movement pattern, temporal, multivariate, statistical, and fatigue-injury interaction).
    
    # Parameters:
    #   - debug (bool): Controls verbose debug output.
    #   - csv_path (str): Path to input CSV file.
    #   - json_path (str): Path to participant info JSON.
    # """
    csv_path="../../data/processed/final_granular_dataset.csv"
    json_path="../../data/basketball/freethrow/participant_information.json"
    data, trial_summary_data, shot_phase_summary_data = prepare_base_datasets(csv_path, json_path, debug=debug)
        

    # data = add_simulated_player_metrics(data, window=5, debug=debug)

    # === NEW ADDITION: Validate Target Columns ===
    try:
        # 'by_trial_exhaustion_score' should be continuous
        validate_target_column(data['by_trial_exhaustion_score'].values, 
                               'by_trial_exhaustion_score', 'continuous')
        # 'injury_risk' should be binary
        validate_target_column(data['injury_risk'].values, 
                               'injury_risk', 'binary')
        # 'by_trial_exhaustion_score' should be continuous
        validate_target_column(trial_summary_data['by_trial_exhaustion_score'].values, 
                               'by_trial_exhaustion_score', 'continuous')
        # 'injury_risk' should be binary
        validate_target_column(trial_summary_data['injury_risk'].values, 
                               'injury_risk', 'binary')
        # 'by_trial_exhaustion_score' should be continuous
        validate_target_column(shot_phase_summary_data['by_trial_exhaustion_score'].values, 
                               'by_trial_exhaustion_score', 'continuous')
        # 'injury_risk' should be binary
        validate_target_column(shot_phase_summary_data['injury_risk'].values, 
                               'injury_risk', 'binary')
    except ValueError as e:
        logging.error(f"Target validation error: {e}")
        sys.exit(1)


    # For demonstration, define features/targets (you can adjust these as needed)
    features_exhaustion = [
        'joint_power', 
        'joint_energy', 
        'elbow_asymmetry',  
        'L_WRIST_angle', 'R_WRIST_angle',  # Updated: removed "wrist_angle"
        'exhaustion_lag1', 
        'power_avg_5',
        'simulated_HR',
        'player_height_in_meters',
        'player_weight__in_kg'
    ]
    target_exhaustion = 'by_trial_exhaustion_score'

    features_injury = [
        'joint_power', 
        'joint_energy', 
        'elbow_asymmetry',  
        'knee_asymmetry', 
        'L_WRIST_angle', 'R_WRIST_angle',  # Updated: removed "wrist_angle"
        'exhaustion_lag1', 
        'power_avg_5',
        'simulated_HR',
        'player_height_in_meters',
        'player_weight__in_kg'
    ]
    target_injury = 'injury_risk'

    
    # Identify joint energy columns (excluding the aggregated 'joint_energy')
    joint_energy_columns = [
        col for col in data.columns
        if '_energy' in col and not ('by_trial' in col or 'overall' in col) and col != 'joint_energy'
    ]
    logging.info(f"Joint Energy Columns after excluding 'joint_energy' ({len(joint_energy_columns)}): {joint_energy_columns}")
    
    # # Execute analysis functions
    # joint_specific_analysis(data, joint_energy_columns, debug=debug)
    # movement_pattern_analysis(data, debug=debug)
    # temporal_analysis_enhancements(data, debug=debug)
    # multivariate_analysis(data, joint_energy_columns, debug=debug)
    # statistical_testing(data, joint_energy_columns, debug=debug)
    # Drop rows with NaNs in the key columns (e.g., energy_acceleration and exhaustion_rate)

    #check if energy_acceleration and exhaustion_rate are dropped
    print("Columns in the final data:", data.columns.tolist())
    print("Number of NaNs in energy_acceleration:", data['energy_acceleration'].isna().sum())
    print("Number of NaNs in exhaustion_rate:", data['exhaustion_rate'].isna().sum())
    logging.info("Processing pipeline completed successfully.")
    
    
    

    # right after feature_engineering(data, ...) or wherever both columns exist:

    import matplotlib.pyplot as plt

    # choose the joint‐specific exhaustion rate you want to check:
    x = data['joint_power']
    y = data['exhaustion_rate']  # or any other '<side>_<JOINT>_exhaustion_rate'

    plt.figure(figsize=(8,6))
    plt.scatter(x, y, alpha=0.6)
    plt.xlabel('Total Joint Power')
    plt.ylabel('L Elbow Exhaustion Rate')
    plt.title('Scatter: Joint Power vs. L Elbow Exhaustion Rate')
    plt.grid(True)
    plt.show()

    data[['L_ELBOW_energy_by_trial_exhaustion_score','by_trial_time']].head(20)


    # ------------------------------------------------------------
    # Example usage with your DataFrame and “x”, “y” columns:
    # ------------------------------------------------------------
    # Suppose `data` is the DataFrame from your pipeline.


    columns_to_check = ["joint_power", "exhaustion_rate"]
    data, dist_results = check_distribution_and_zscore(data, columns_to_check)

    print("\before scaling z-score columns added to DataFrame:")
    print([col for col in data.columns if col.endswith('_zscore')])

    print("\nbefore scaling Distribution check summary:")
    print(dist_results)
    
    # Suppose we want to scale just these two columns for baseline LSTM:
    columns_to_scale = ["joint_power", "exhaustion_rate"]

    # 1) Use StandardScaler (z-score) as a baseline:
    scaled_data, standard_scaler = scale_for_lstm(
        df=data,
        columns_to_scale=columns_to_scale,
        method="standard",  # or "minmax"
        debug=True
    )
    
    columns_to_check = ["joint_power", "exhaustion_rate"]
    data, dist_results = check_distribution_and_zscore(scaled_data, columns_to_check)

    print("\nFinal z-score columns added to DataFrame:")
    print([col for col in data.columns if col.endswith('_zscore')])

    print("\nDistribution check summary:")
    print(dist_results)
