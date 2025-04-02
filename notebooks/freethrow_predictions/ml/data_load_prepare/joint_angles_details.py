import numpy as np
import pandas as pd
import logging

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

def calculate_joint_angles_over_motion(df, release_frame_index, debug=False):
    """
    Calculate joint angles over motion for both left and right sides.
    
    For each side ('L' and 'R'), the function computes joint angles using the defined joint combinations,
    creates ongoing angle columns, and then calculates the max and release angles (using the shooting motion rows).
    
    Joint names are set in UPPERCASE (ELBOW, WRIST, KNEE) for consistency.
    
    Parameters:
      - df: DataFrame containing motion data.
      - release_frame_index: Index for the release point.
      - debug: If True, prints debug information.
    
    Returns:
      - df: Updated DataFrame with joint angle columns and new ongoing angle columns for both sides.
      - joint_angle_metrics_df: DataFrame (single row) with max and release angles for the specified joints for both sides.
    """
    # Define the sides to process
    sides = ['L', 'R']
    
    # Dictionary to store joint metrics for both sides
    overall_joint_angle_metrics = {}
    
    # Process each side separately
    for side in sides:
        # Define joint combinations for the given side with joint names in UPPERCASE
        joint_combinations = {
            'ELBOW': [f'{side}_SHOULDER', f'{side}_ELBOW', f'{side}_WRIST'],
            'WRIST': [f'{side}_ELBOW', f'{side}_WRIST', f'{side}_1STFINGER'],
            'KNEE': [f'{side}_HIP', f'{side}_KNEE', f'{side}_ANKLE']
        }
        
        # Step 1: Calculate joint angles for all rows and store them in new columns
        for joint_name, (a, b, c) in joint_combinations.items():
            # The new column will be named as "<side>_<joint_name>_angle", e.g., "L_ELBOW_angle"
            angle_col = f'{side}_{joint_name}_angle'
            df[angle_col] = df.apply(
                lambda row: calculate_joint_angle(
                    row[[f"{a}_x", f"{a}_y", f"{a}_z"]].values,
                    row[[f"{b}_x", f"{b}_y", f"{b}_z"]].values,
                    row[[f"{c}_x", f"{c}_y", f"{c}_z"]].values
                ),
                axis=1
            )
            if debug:
                print(f"Debug: Calculated {side} {joint_name} angles (column '{angle_col}'):\n", df[[angle_col]].head())
        
        # --- New Feature: Add ongoing angle columns with side indicator ---
        # Create ongoing angle columns named as "<side>_<joint_name>_ongoing_angle"
        for joint_name in joint_combinations.keys():
            ongoing_col = f'{side}_{joint_name}_ongoing_angle'
            # Copy the calculated angle into the ongoing column
            df[ongoing_col] = df[f'{side}_{joint_name}_angle']
            if debug:
                print(f"Debug: Added ongoing angle column for {side} {joint_name}: '{ongoing_col}'")
        
        # --- Calculate Joint Angle Metrics for Shooting Motion ---
        # Filter for shooting motion rows (assumes shooting_motion==1 indicates shooting frames)
        shooting_motion_df = df[df['shooting_motion'] == 1]
        
        # For each joint, compute the max and release angles
        for joint_name in joint_combinations.keys():
            angle_col = f'{side}_{joint_name}_angle'
            # Calculate the maximum angle within shooting motion
            max_angle = shooting_motion_df[angle_col].max()
            # Get the release angle from the frame at release_frame_index (if present)
            if release_frame_index in shooting_motion_df.index:
                release_angle = shooting_motion_df.at[release_frame_index, angle_col]
            else:
                release_angle = np.nan
            # Store metrics with keys that include the side indicator and uppercase joint name
            overall_joint_angle_metrics[f'{side}_{joint_name}_max_angle'] = max_angle
            overall_joint_angle_metrics[f'{side}_{joint_name}_release_angle'] = release_angle
            if debug:
                print(f"Debug: {side} {joint_name} - max_angle: {max_angle}, release_angle: {release_angle}")
    
    # Step 5: Convert metrics dictionary to a single-row DataFrame for output
    joint_angle_metrics_df = pd.DataFrame([overall_joint_angle_metrics])
    
    # Final debug: show updated DataFrame and metrics
    if debug:
        print("Debug: Main DataFrame with ongoing joint angles (first few rows):\n", df.head())
        print("Debug: Joint Angle Metrics DataFrame (single row):\n", joint_angle_metrics_df)
    
    return df, joint_angle_metrics_df


def split_shooting_phases_dynamic(data, handedness="R", debug=False):
    """
    Dynamically splits the basketball shooting motion into four detailed phases based on biomechanical events.
    
    The intended phases are:
      - leg_cock: From the start of the shooting motion until the frame where the average knee angle 
                  (from L_KNEE_angle and R_KNEE_angle) is maximum.
      - arm_cock: From immediately after the leg_cock phase until the frame where the average elbow angle 
                  (from L_ELBOW_angle and R_ELBOW_angle) is maximum.
      - arm_release: From immediately after the arm_cock phase until the first frame where 
                     'release_point_filter' is True.
      - wrist_release: From immediately after the release event until the frame (even beyond the contiguous shooting motion)
                       where the elbow goes below the shoulder level.
    
    Note: The post_release phase has been removed per the new specifications.
    
    For each shooting sequence within each trial (where shooting_motion == 1), the function computes:
      1. The leg event as the frame with maximum average knee angle.
      2. The arm event as the frame (after the leg event) with maximum average elbow angle.
      3. The release event as the first frame (after the arm event) where release_point_filter is True.
      4. The wrist event as the first frame (after the release event, searching the full trial)
         where the elbow is below the shoulder level.
    
    Additionally, for each shooting sequence, the computed event indices are stored in new columns:
         'event_idx_leg', 'event_idx_elbow', 'event_idx_release', 'event_idx_wrist'
    
    Parameters:
      - data (pd.DataFrame): DataFrame expected to contain:
           'shooting_motion', 'L_KNEE_ongoing_angle', 'R_KNEE_ongoing_angle',
           'L_ELBOW_ongoing_angle', 'R_ELBOW_ongoing_angle', 'release_point_filter',
           'R_WRIST_ongoing_angle',
           and for the corresponding joint position columns: 
           For right-handed: 'R_ELBOW_y', 'R_SHOULDER_y'
           For left-handed (if handedness=="L"): 'L_ELBOW_y', 'L_SHOULDER_y'
      - handedness (str): 'R' or 'L' indicating right or left hand (default is "R").
      - debug (bool): If True, prints detailed debug information.
    
    Returns:
      - pd.DataFrame: The input DataFrame with an added column 'shooting_phases' that labels each frame 
                      in a shooting sequence and new columns for event indices.
    """
    import logging
    import numpy as np

    # --- Preliminary Checks ---
    required_cols = ['shooting_motion', 'L_KNEE_ongoing_angle', 'R_KNEE_ongoing_angle',
                     'L_ELBOW_ongoing_angle', 'R_ELBOW_ongoing_angle', 'release_point_filter',
                     'R_WRIST_ongoing_angle', 'R_ELBOW_y', 'R_SHOULDER_y']
    if handedness == "L":
        for col in ['L_ELBOW_y', 'L_SHOULDER_y']:
            if col not in data.columns:
                logging.warning(f"Required column '{col}' not found for left-handed wrist event calculation. Skipping dynamic shooting phase split.")
                logging.info(f"Data columns: {data.columns.tolist()}")
                return data

    for col in required_cols:
        if col not in data.columns:
            logging.warning(f"Required column '{col}' not found. Skipping dynamic shooting phase split.")
            logging.info(f"Data columns: {data.columns.tolist()}")
            return data

    # --- Initialize Columns ---
    if 'shooting_phases' not in data.columns:
        data['shooting_phases'] = np.nan
        data['shooting_phases'] = data['shooting_phases'].astype(object)

    data['event_idx_leg'] = np.nan
    data['event_idx_elbow'] = np.nan
    data['event_idx_release'] = np.nan
    data['event_idx_wrist'] = np.nan

    if debug:
        indices = data.index.tolist()
        logging.info(f"Data index range: {indices[0]} to {indices[-1]}, total rows: {len(indices)}")
        logging.info(f"First 10 indices: {indices[:10]}")
        logging.info(f"Last 10 indices: {indices[-10:]}")
        if 'release_point_filter' in data.columns:
            rp_counts = data['release_point_filter'].value_counts(dropna=False)
            logging.info(f"'release_point_filter' value counts:\n{rp_counts}")
            logging.info("Sample of 'release_point_filter' values (first 20 rows):")
            logging.info(data['release_point_filter'].head(20).to_string())

    # --- Log Unique Trial IDs ---
    if 'trial_id' in data.columns:
        unique_trials = data['trial_id'].unique()
        logging.info(f"Unique trial IDs: {unique_trials}")
    else:
        unique_trials = [None]

    # --- Process Each Trial Using Groupby ---
    # This ensures that each trial is processed separately.
    for trial, trial_data in data.groupby('trial_id'):
        if debug:
            logging.info(f"Processing trial {trial} with {len(trial_data)} rows.")
        # Find shooting indices for the current trial
        trial_shooting_idx = trial_data.index[trial_data['shooting_motion'] == 1].tolist()
        if debug:
            logging.info(f"Trial {trial}: Found {len(trial_shooting_idx)} shooting frames.")

        # Group contiguous shooting indices within the trial
        sequences = []
        if trial_shooting_idx:
            seq = [trial_shooting_idx[0]]
            for idx in trial_shooting_idx[1:]:
                if idx == seq[-1] + 1:
                    seq.append(idx)
                else:
                    sequences.append(seq)
                    seq = [idx]
            sequences.append(seq)
        if debug:
            logging.info(f"Trial {trial}: Grouped into {len(sequences)} shooting sequences.")

        # --- Process Each Shooting Sequence ---
        phase_labels = ['leg_cock', 'arm_cock', 'arm_release', 'wrist_release']
        for seq in sequences:
            n = len(seq)
            if n < 4:
                if debug:
                    logging.info(f"Trial {trial}: Skipping sequence {seq} due to insufficient length ({n}).")
                continue

            logging.info(f"\nProcessing shooting sequence for trial {trial} starting at index {seq[0]} ending at {seq[-1]} (total frames: {n}).")
            logging.info(f"Sequence frame indices: {seq}")

            seq_data = data.loc[seq]

            # --- Compute Leg Event ---
            knee_avg = seq_data[['L_KNEE_angle', 'R_KNEE_angle']].mean(axis=1)
            idx_leg = knee_avg.idxmax()
            logging.info(f"Leg event: Maximum average knee angle = {knee_avg.loc[idx_leg]:.2f} at index {idx_leg}.")

            # --- Compute Arm Event ---
            remaining1 = [i for i in seq if i > idx_leg]
            if remaining1:
                elbow_avg = data.loc[remaining1, ['L_ELBOW_angle', 'R_ELBOW_angle']].mean(axis=1)
                idx_elbow = elbow_avg.idxmax()
                logging.info(f"Arm event: Maximum average elbow angle = {elbow_avg.loc[idx_elbow]:.2f} at index {idx_elbow}.")
            else:
                idx_elbow = None
                logging.info("No frames available after leg event for arm event.")

            # --- Compute Release Event ---
            if idx_elbow is not None:
                remaining2 = [i for i in remaining1 if i > idx_elbow]
            else:
                remaining2 = remaining1
            if remaining2:
                rp_values = data.loc[remaining2, 'release_point_filter'].tolist()
                logging.info(f"Release candidate range indices: {remaining2}; release_point_filter values: {rp_values}")
                release_found = data.loc[remaining2][data.loc[remaining2, 'release_point_filter'] == True]
                if not release_found.empty:
                    idx_release = release_found.index[0]
                    logging.info(f"Release event: 'release_point_filter' is True at index {idx_release}.")
                else:
                    idx_release = remaining2[-1]
                    logging.info(f"Release event: No True value found; using last candidate index {idx_release}.")
            else:
                idx_release = None
                logging.info("No frames available after arm event for release event.")

            # --- Compute Wrist Event (Extended Search Across the Trial) ---
            if idx_release is not None:
                # Search the full trial for the wrist event condition
                candidate = trial_data[trial_data.index > idx_release]
                if handedness == 'L':
                    condition = candidate['L_ELBOW_y'] > candidate['L_SHOULDER_y']
                else:
                    condition = candidate['R_ELBOW_y'] > candidate['R_SHOULDER_y']
                wrist_candidates = candidate[condition]
                if not wrist_candidates.empty:
                    idx_wrist = wrist_candidates.index[0]
                    logging.info(f"Wrist event: Elbow below shoulder detected at index {idx_wrist}.")
                else:
                    idx_wrist = None
                    logging.info("No frame found where elbow is below shoulder after release event for wrist event.")
            else:
                idx_wrist = None
                logging.info("No frames available after release event for wrist event.")

            logging.info(f"Computed event indices: idx_leg={idx_leg}, idx_elbow={idx_elbow}, idx_release={idx_release}, idx_wrist={idx_wrist}")

            # --- New Detailed Debugging Report ---
            # This block provides an in-depth, structured report of the shooting motion events.
            if debug:
                shooting_start, shooting_end = seq[0], seq[-1]
                report_message = "\n--- Detailed Debugging Report for Trial {} ---\n".format(trial)
                report_message += f"Shooting motion range: [{shooting_start} to {shooting_end}] (Total frames: {n})\n"
                
                # Helper function: Gives a readable frame info relative to the start of the sequence.
                def frame_relative(idx):
                    return f"{idx} (Frame {idx - shooting_start + 1} of {n})" if idx is not None else "Not Found"
                
                # Report each event using the helper function.
                report_message += f"1. Leg event (max avg knee angle): {frame_relative(idx_leg)}\n"
                report_message += f"2. Arm event (max avg elbow angle after leg): {frame_relative(idx_elbow)}\n"
                report_message += f"3. Release event (first release_point_filter=True after arm): {frame_relative(idx_release)}\n"
                report_message += f"4. Wrist event (elbow below shoulder after release): {frame_relative(idx_wrist)}\n"
                
                # Check for release_point_filter=True positions after the arm event.
                if idx_elbow is not None:
                    rp_candidates = [i for i in seq if i > idx_elbow and data.loc[i, 'release_point_filter']]
                    if rp_candidates:
                        report_message += f"   → release_point_filter=True found at indices: {rp_candidates}\n"
                    else:
                        report_message += "   → No release_point_filter=True found after arm event within shooting motion.\n"
                
                # Validate the sequential order of the events.
                issues = []
                if idx_leg is None:
                    issues.append("Leg event missing")
                if idx_elbow is None or (idx_leg is not None and idx_elbow <= idx_leg):
                    issues.append("Arm event missing or before leg event")
                if idx_release is None or (idx_elbow is not None and idx_release <= idx_elbow):
                    issues.append("Release event missing or before arm event")
                if idx_wrist is None or (idx_release is not None and idx_wrist <= idx_release):
                    issues.append("Wrist event missing or before release event")
                
                if issues:
                    report_message += f"❗ Issues Detected: {', '.join(issues)}\n"
                else:
                    report_message += "✅ All events detected in correct order.\n"
                
                logging.info(report_message)

            # --- Build Candidate Segments for Phases ---
            seg_leg = [i for i in seq if i <= idx_leg] if idx_leg is not None else []
            seg_arm = [i for i in seq if idx_leg is not None and i > idx_leg and (idx_elbow is None or i <= idx_elbow)]
            seg_release = [i for i in seq if idx_elbow is not None and i > idx_elbow and (idx_release is None or i <= idx_release)]
            # For wrist_release, extend the segment if it is very short.
            if idx_release is not None:
                trial_indices = trial_data.index.tolist()
                if idx_wrist is not None:
                    seg_wrist = [i for i in trial_indices if i > idx_release and i <= idx_wrist]
                    if len(seg_wrist) < 3:
                        seg_wrist = [i for i in trial_indices if i > idx_release]
                        logging.info(f"Extended wrist_release candidate segment for trial {trial} because original segment was short ({len(seg_wrist)} frames).")
                else:
                    seg_wrist = [i for i in trial_indices if i > idx_release]
            else:
                seg_wrist = []
            candidate_segments = [seg_leg, seg_arm, seg_release, seg_wrist]

            # --- Log Candidate Segment Details ---
            for label, seg in zip(phase_labels, candidate_segments):
                if seg:
                    logging.info(f"Candidate segment for phase '{label}' has {len(seg)} frames: from index {seg[0]} to {seg[-1]}.")
                else:
                    logging.info(f"Candidate segment for phase '{label}' is empty.")

            # Check that all phases have at least one frame.
            valid = all(len(seg) > 0 for seg in candidate_segments)
            if not valid:
                error_msg = (f"Error: Candidate segmentation did not yield all phases for trial {trial} "
                             f"sequence starting at index {seq[0]}. Candidate segments lengths: {[len(seg) for seg in candidate_segments]}")
                raise ValueError(error_msg)

            # --- Assign Phase Labels ---
            for phase, indices in zip(phase_labels, candidate_segments):
                for i in indices:
                    data.loc[i, 'shooting_phases'] = phase

            # --- Assign Computed Event Indices ---
            data.loc[seq, 'event_idx_leg'] = idx_leg
            data.loc[seq, 'event_idx_elbow'] = idx_elbow
            data.loc[seq, 'event_idx_release'] = idx_release
            data.loc[seg_wrist, 'event_idx_wrist'] = idx_wrist

            if debug:
                phase_distribution = trial_data['shooting_phases'].value_counts(dropna=False).to_dict()
                logging.info(f"Trial {trial}: Assigned phase labels. Phase distribution: {phase_distribution}")
    
    return data







if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Debug: Loading and parsing file: ../../SPL-Open-Data/basketball/freethrow/data/P0001/BB_FT_P0001_T0001.json")
    from ml.data_load_prepare.load_and_parse import load_single_ft_and_parse
    from ml.data_load_prepare.dataframe_creation import main_create_dataframe
    from ml.data_load_prepare.velocity_and_speed_calc import calculate_ball_speed_velocity_direction
    from ml.data_load_prepare.phase_labeling import main_label_shot_phases
    from ml.data_load_prepare.ball_trajectory_and_release_time_stats import main_ball_trajectory_analysis
    from ml.data_load_prepare.joint_power_calc import main_calculate_joint_power

    # Load and process data
    data, trial_id, result, landing_x, landing_y, entry_angle, _ = load_single_ft_and_parse(
        "../../SPL-Open-Data/basketball/freethrow/data/P0001/BB_FT_P0001_T0002.json"
    )
    debug = True
    df = main_create_dataframe(data, trial_id, result, landing_x, landing_y, entry_angle, debug=False)
    df = calculate_ball_speed_velocity_direction(df, debug=False)
    df = main_label_shot_phases(df)
    release_frame_index = df.index[df['release_point_filter'] == 1].tolist()[0]
    df, release_index, projection_df, ml_metrics_df = main_ball_trajectory_analysis(df, release_frame_index, debug=False)
    
    # Test the joint angles function with ongoing angles
    df, joint_angle_metrics_df = calculate_joint_angles_over_motion(df, release_frame_index, debug=True)
    print("Final DataFrame with ongoing joint angles across all rows:\n", df.head())
    print("Joint Angle Metrics DataFrame (max and release angles during shooting motion):\n", joint_angle_metrics_df)
    
    # ---------------------------
    # Additional Debug: Compute and log key event indices per trial
    # ---------------------------
    # For each trial, we log:
    # - Full trial indices
    # - Shooting_motion indices (where shooting_motion == 1)
    # - Shooting_motion segment start/end
    # - Computed event indices (leg, arm, release, wrist)
    print("Unique trials ids===========", df['trial_id'].unique())
    for trial, trial_data in df.groupby('trial_id'):
        logging.info(f"\n--- Debug for Trial {trial} ---")
        full_indices = trial_data.index.tolist()
        logging.info(f"Full trial indices: {full_indices}")
        
        # Shooting_motion indices (subset where shooting_motion == 1)
        shooting_indices = trial_data.index[trial_data['shooting_motion'] == 1].tolist()
        logging.info(f"Shooting_motion indices: {shooting_indices}")
        
        if shooting_indices:
            logging.info(f"Shooting_motion begins at index {shooting_indices[0]} and ends at index {shooting_indices[-1]}")
            
            # Compute key event indices using the full trial (for leg and wrist events)
            full_knee_avg = trial_data[['L_KNEE_angle', 'R_KNEE_angle']].mean(axis=1)
            full_idx_leg = full_knee_avg.idxmax()
            logging.info(f"Full trial leg event candidate index (max knee angle): {full_idx_leg}")
            
            # Compute key event indices on the shooting_motion subset
            seq = shooting_indices
            seq_data = trial_data.loc[seq]
            knee_avg = seq_data[['L_KNEE_angle', 'R_KNEE_angle']].mean(axis=1)
            idx_leg = knee_avg.idxmax()
            logging.info(f"Shooting_motion leg event index: {idx_leg} (knee angle = {knee_avg.loc[idx_leg]:.2f})")
            
            # Compute Arm Event on shooting_motion subset (frames after leg event)
            remaining1 = [i for i in seq if i > idx_leg]
            if remaining1:
                elbow_avg = trial_data.loc[remaining1, ['L_ELBOW_angle', 'R_ELBOW_angle']].mean(axis=1)
                idx_elbow = elbow_avg.idxmax()
                logging.info(f"Shooting_motion arm event index: {idx_elbow} (elbow angle = {elbow_avg.loc[idx_elbow]:.2f})")
            else:
                idx_elbow = None
                logging.info("No frames available after leg event for arm event.")
            
            # Compute Release Event on shooting_motion subset (frames after arm event)
            if idx_elbow is not None:
                remaining2 = [i for i in remaining1 if i > idx_elbow]
            else:
                remaining2 = remaining1
            if remaining2:
                rp_values = trial_data.loc[remaining2, 'release_point_filter'].tolist()
                logging.info(f"Release candidate range indices: {remaining2}; release_point_filter values: {rp_values}")
                release_found = trial_data.loc[remaining2][trial_data.loc[remaining2, 'release_point_filter'] == True]
                if not release_found.empty:
                    idx_release = release_found.index[0]
                    logging.info(f"Shooting_motion release event index: {idx_release} (release_point_filter is True)")
                else:
                    idx_release = remaining2[-1]
                    logging.info(f"Shooting_motion: No True release_point_filter found; using last candidate index: {idx_release}")
            else:
                idx_release = None
                logging.info("No frames available after arm event for release event.")
            
            # Compute Wrist Event using the full trial (search frames after release event)
            if idx_release is not None:
                candidate = trial_data[trial_data.index > idx_release]
                # Adjust condition based on handedness; assuming handedness "R" here
                if "handedness" not in globals():
                    handedness = "R"
                if handedness == 'L':
                    condition = candidate['L_ELBOW_y'] > candidate['L_SHOULDER_y']
                else:
                    condition = candidate['R_ELBOW_y'] > candidate['R_SHOULDER_y']
                wrist_candidates = candidate[condition]
                if not wrist_candidates.empty:
                    idx_wrist = wrist_candidates.index[0]
                    logging.info(f"Full trial wrist event index (first frame where elbow below shoulder): {idx_wrist}")
                else:
                    idx_wrist = None
                    logging.info("No wrist event candidate found in full trial after release event.")
            else:
                idx_wrist = None
                logging.info("Wrist event not computed because no release event was found.")
            
            logging.info(f"Computed indices for Trial {trial}: leg: {idx_leg}, arm: {idx_elbow}, release: {idx_release}, wrist: {idx_wrist}")
        else:
            logging.info(f"Trial {trial} has no shooting_motion frames.")
    
    # ---------------------------
    # Continue with phase labeling
    # ---------------------------
    df = split_shooting_phases_dynamic(df, handedness="R", debug=False)
    
    # Group by 'trial_id' and 'shooting_phases' to get frame counts per phase per trial
    phase_counts = df.groupby(['trial_id', 'shooting_phases']).size().reset_index(name='frame_count')
    print("Frame counts per phase per trial:")
    print(phase_counts)
    
    # Create a pivot table: trials as rows, phases as columns, and frame counts as values
    pivot_phase_counts = phase_counts.pivot(index='trial_id', columns='shooting_phases', values='frame_count')
    print("\nPivot table of frame counts per trial and phase:")
    print(pivot_phase_counts)
    
    # Check if each trial has all expected phases
    expected_phases = ['arm_cock', 'arm_release', 'leg_cock', 'wrist_release']
    # For each trial, count the number of non-null phases
    phase_availability = pivot_phase_counts[expected_phases].notnull().sum(axis=1)
    print("\nNumber of available phases per trial (out of expected phases):")
    print(phase_availability)
    
    # Optionally, list trials that don't have all expected phases
    trials_missing = phase_availability[phase_availability < len(expected_phases)]
    print("\nTrials missing some shooting phases:")
    print(trials_missing)
  
    
    # metrics df's for machine learning dataset and base dataset
    # - joint_power_metrics_df: DataFrame with joint power metrics calculated during shooting motion
    # - joint_angle_metrics_df (max and release angles during shooting motion)
    # - shot_details_df = df[['result', 'landing_x', 'landing_y', 'entry_angle']].drop_duplicates()
    # ^ shot_details_df are the base metrics from the dataset and the y variable = result
