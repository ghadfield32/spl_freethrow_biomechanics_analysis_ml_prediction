import pandas as pd
import numpy as np

def calculate_release_angles(df, handedness='R', debug=False):
    """
    Calculate the release angles for knee, wrist, and elbow at the release point.
    Adds the calculated angles as new columns to the DataFrame.
    Parameters:
    - df: DataFrame containing the angle data.
    - handedness: 'R' for right-handed, 'L' for left-handed.
    - debug: If True, prints detailed outputs at each step.
    Returns:
    - df: Updated DataFrame with release angles added.
    """
    # Check if the handedness is valid
    if handedness not in ['R', 'L']:
        raise ValueError("Handedness must be 'R' for right-handed or 'L' for left-handed.")
    if debug:
        print(f"Calculating release angles for {handedness}-handed shots.")
    # Calculate angles based on handedness
    if handedness == 'R':
        df['knee_angle'] = df['R_KNEE_angle']
        df['wrist_angle'] = df['R_WRIST_angle']
        df['elbow_angle'] = df['R_ELBOW_angle']
    else:
        df['knee_angle'] = df['L_KNEE_angle']
        df['wrist_angle'] = df['L_WRIST_angle']
        df['elbow_angle'] = df['L_ELBOW_angle']
    release_df = df[df['release_point_filter'] == 1]

    # Calculate release angles for knee, wrist, and elbow
    release_angles = release_df.groupby('trial_id').agg({
        'knee_angle': 'mean',
        'wrist_angle': 'mean',
        'elbow_angle': 'mean'
    }).rename(columns={
        'knee_angle': 'release_knee_angle',
        'wrist_angle': 'release_wrist_angle',
        'elbow_angle': 'release_elbow_angle'
    })

    # Merge release angles back onto the original DataFrame
    df = df.merge(release_angles, on='trial_id', how='left')

    return df
    
    
    

def calculate_optimal_release_angle_ranges(df, debug=False, lower_percentile=10, upper_percentile=90):
    """
    Calculate the initial and filtered optimal min and max ranges for knee, wrist, and elbow angles at the release point.
    Adds the calculated ranges as columns to every row in the DataFrame, and classifies shot quality for each joint.

    Parameters:
    - df: DataFrame containing the angle data with 'release_point_filter' column set to 1 for relevant rows.
    - debug: If True, prints detailed outputs at each step.
    - lower_percentile: The lower percentile threshold for filtering outliers (default is 10).
    - upper_percentile: The upper percentile threshold for filtering outliers (default is 90).

    Returns:
    - df: Updated DataFrame with optimal angle range columns and classifications added to every row.
    """
    # Filter the DataFrame for rows where release_point_filter == 1 and result == 1 (successful shots)
    release_df = df[(df['release_point_filter'] == 1) & (df['result'] == 1)]
    
    # Initialize the new columns to NaN in the main DataFrame
    for col in ['knee_release_angle_initial_optimal_min', 'knee_release_angle_initial_optimal_max',
                'knee_release_angle_filtered_optimal_min', 'knee_release_angle_filtered_optimal_max',
                'wrist_release_angle_initial_optimal_min', 'wrist_release_angle_initial_optimal_max',
                'wrist_release_angle_filtered_optimal_min', 'wrist_release_angle_filtered_optimal_max',
                'elbow_release_angle_initial_optimal_min', 'elbow_release_angle_initial_optimal_max',
                'elbow_release_angle_filtered_optimal_min', 'elbow_release_angle_filtered_optimal_max']:
        df[col] = float('nan')
    
    if not release_df.empty:
        # Calculate initial min and max values (before removing outliers)
        initial_ranges = {
            'knee_release_angle_initial_optimal_min': release_df['knee_angle'].min(),
            'knee_release_angle_initial_optimal_max': release_df['knee_angle'].max(),
            'wrist_release_angle_initial_optimal_min': release_df['wrist_angle'].min(),
            'wrist_release_angle_initial_optimal_max': release_df['wrist_angle'].max(),
            'elbow_release_angle_initial_optimal_min': release_df['elbow_angle'].min(),
            'elbow_release_angle_initial_optimal_max': release_df['elbow_angle'].max()
        }
        
        if debug:
            print("Initial Angle Ranges (Before Removing Outliers):")
            for key, value in initial_ranges.items():
                print(f"- {key}: {value:.2f}°")
        
        # Calculate the specified percentiles for each angle
        knee_angle_lower = release_df['knee_angle'].quantile(lower_percentile / 100.0)
        knee_angle_upper = release_df['knee_angle'].quantile(upper_percentile / 100.0)
        wrist_angle_lower = release_df['wrist_angle'].quantile(lower_percentile / 100.0)
        wrist_angle_upper = release_df['wrist_angle'].quantile(upper_percentile / 100.0)
        elbow_angle_lower = release_df['elbow_angle'].quantile(lower_percentile / 100.0)
        elbow_angle_upper = release_df['elbow_angle'].quantile(upper_percentile / 100.0)
        
        # Filter out outliers based on the specified percentile range for each angle
        filtered_release_df = release_df[
            (release_df['knee_angle'] >= knee_angle_lower) & (release_df['knee_angle'] <= knee_angle_upper) &
            (release_df['wrist_angle'] >= wrist_angle_lower) & (release_df['wrist_angle'] <= wrist_angle_upper) &
            (release_df['elbow_angle'] >= elbow_angle_lower) & (release_df['elbow_angle'] <= elbow_angle_upper)
        ]
        
        # Calculate min and max values after filtering
        filtered_ranges = {
            'knee_release_angle_filtered_optimal_min': filtered_release_df['knee_angle'].min(),
            'knee_release_angle_filtered_optimal_max': filtered_release_df['knee_angle'].max(),
            'wrist_release_angle_filtered_optimal_min': filtered_release_df['wrist_angle'].min(),
            'wrist_release_angle_filtered_optimal_max': filtered_release_df['wrist_angle'].max(),
            'elbow_release_angle_filtered_optimal_min': filtered_release_df['elbow_angle'].min(),
            'elbow_release_angle_filtered_optimal_max': filtered_release_df['elbow_angle'].max()
        }
        
        if debug:
            print(f"\nFiltered Angle Ranges (After Removing {lower_percentile}th and {upper_percentile}th Percentile Outliers):")
            for key, value in filtered_ranges.items():
                print(f"- {key}: {value:.2f}°")

        # Assign these values to all rows in the main DataFrame
        for key, value in {**initial_ranges, **filtered_ranges}.items():
            df[key] = value
        
        # Step 6: Add classification for each joint
        def classify_joint(angle_value, min_val, max_val):
            if angle_value < min_val:
                return "Early"
            elif angle_value > max_val:
                return "Late"
            else:
                return "Good"
        
        # Apply classification to merged_df
        df['wrist_release_angle_shot_classification'] = df.apply(
            lambda row: classify_joint(row['wrist_angle'], filtered_ranges['wrist_release_angle_filtered_optimal_min'], filtered_ranges['wrist_release_angle_filtered_optimal_max']),
            axis=1
        )
        df['elbow_release_angle_shot_classification'] = df.apply(
            lambda row: classify_joint(row['elbow_angle'], filtered_ranges['elbow_release_angle_filtered_optimal_min'], filtered_ranges['elbow_release_angle_filtered_optimal_max']),
            axis=1
        )
        df['knee_release_angle_shot_classification'] = df.apply(
            lambda row: classify_joint(row['knee_angle'], filtered_ranges['knee_release_angle_filtered_optimal_min'], filtered_ranges['knee_release_angle_filtered_optimal_max']),
            axis=1
        )
        
        if debug:
            print("\nShot Classifications Added:")
            print(df[['trial_id', 'wrist_release_angle_shot_classification', 'elbow_release_angle_shot_classification'
                      , 'knee_release_angle_shot_classification'
                      , 'knee_release_angle_filtered_optimal_min', 'knee_release_angle_filtered_optimal_max'
                      , 'elbow_release_angle_filtered_optimal_min', 'elbow_release_angle_filtered_optimal_max'
                      , 'wrist_release_angle_filtered_optimal_min', 'wrist_release_angle_filtered_optimal_max']].head())
        
    else:
        if debug:
            print("No rows found where release_point_filter is set to 1.")
    
    return df



def calculate_optimal_max_angle_ranges(final_granular_df, output_path, debug=False, lower_percentile=10, upper_percentile=90):
    """
    Calculates min and max wrist, elbow, and knee angles from successful shots during shooting motion,
    while applying a filter to remove outliers based on specified percentiles. The function then
    adds these statistics as new columns to `final_granular_df`, classifies each joint's shot as Early, Good, or Late,
    and adds binary columns marking the maximum angle point for each joint.
    
    Args:
        final_granular_df (pd.DataFrame): The input DataFrame with shot data.
        debug (bool): Whether to print debug information for each processing step.
        lower_percentile (int): Lower percentile for outlier removal.
        upper_percentile (int): Upper percentile for outlier removal.
    
    Returns:
        pd.DataFrame: `final_granular_df` with additional columns for min and max values of each angle,
                      separate shot classifications for each joint, and binary indicators for maximum angle points.
    """
    
    # Initial debug: DataFrame structure and size
    if debug:
        print("Initial DataFrame length:", len(final_granular_df))
        print("Columns in final_granular_df:\n", final_granular_df.columns)
        total_trials_before_filter = final_granular_df['trial_id'].nunique()
        print(f"\nTotal number of trials before filtering by `result`: {total_trials_before_filter}")
    
    # Step 1: Filter for active shooting motion
    motion_df = final_granular_df[final_granular_df['shooting_motion'] == 1]
    if debug:
        print("\nFiltered DataFrame for rows where shooting motion is active:")
        print("Number of rows after filtering by shooting motion:", len(motion_df))
    
    # Step 2: Calculate max angles per trial for wrist, elbow, and knee
    max_angles_per_trial = motion_df.groupby('trial_id').agg({
        'wrist_angle': 'max',
        'elbow_angle': 'max',
        'knee_angle': 'max'
    }).reset_index()
    
    if debug:
        print("\nCalculated maximum angles per trial during shooting motion:")
        print(max_angles_per_trial.head())
    
    # Step 3: Merge max angles with motion_df to mark the max points
    merged_df = motion_df.merge(
        max_angles_per_trial.rename(columns={
            'wrist_angle': 'wrist_max_angle',
            'elbow_angle': 'elbow_max_angle',
            'knee_angle': 'knee_max_angle'
        }),
        on='trial_id', how='left'
    )
    
    # Create binary indicators for the max angle points
    merged_df['is_wrist_max_angle'] = ((merged_df['wrist_angle'] == merged_df['wrist_max_angle'])).astype(int)
    merged_df['is_elbow_max_angle'] = ((merged_df['elbow_angle'] == merged_df['elbow_max_angle'])).astype(int)
    merged_df['is_knee_max_angle'] = ((merged_df['knee_angle'] == merged_df['knee_max_angle'])).astype(int)
    
    
    # Step 4: Filter for successful shots
    successful_shots_df = merged_df[merged_df['result'] == 1]
    if debug:
        print("\nFiltered DataFrame for successful shots (result=1):")
        print("Number of rows after filtering by `result`:", len(successful_shots_df))
        total_trials_after_result_filter = successful_shots_df['trial_id'].nunique()
        print(f"Total number of trials after filtering by `result`: {total_trials_after_result_filter}")
    
    # Step 5: Calculate initial and filtered stats
    stats = {}
    for angle in ['wrist_max_angle', 'elbow_max_angle', 'knee_max_angle']:
        # Define percentiles for filtering
        lower_bound = np.percentile(successful_shots_df[angle], lower_percentile)
        upper_bound = np.percentile(successful_shots_df[angle], upper_percentile)
    
        # Store filtered min and max in stats
        stats[f"{angle}_initial_optimal_min"] = successful_shots_df[angle].min()
        stats[f"{angle}_initial_optimal_max"] = successful_shots_df[angle].max()
        
        # Filter outliers
        filtered_data = successful_shots_df[(successful_shots_df[angle] >= lower_bound) &
                                            (successful_shots_df[angle] <= upper_bound)]
        
        # Store filtered min and max in stats
        stats[f"{angle}_filtered_optimal_min"] = filtered_data[angle].min()
        stats[f"{angle}_filtered_optimal_max"] = filtered_data[angle].max()
    
    if debug:
        print("\nOptimal knee/wrist/elbow angles filtered by the percentile range:")
        print(stats)
        
    for key, value in stats.items():
        merged_df[key] = value
        
    if debug:
        print("\nOptimal knee/wrist/elbow angles added to the merged df:")
        print(merged_df.columns)
        
    # Step 6: Add classification for each joint
    def classify_joint(angle_value, min_val, max_val):
        if angle_value < min_val:
            return "Early"
        elif angle_value > max_val:
            return "Late"
        else:
            return "Good"

    # Apply classification to merged_df
    merged_df['wrist_max_angle_shot_classification'] = merged_df.apply(
        lambda row: classify_joint(row['wrist_angle'], stats['wrist_max_angle_filtered_optimal_min'], stats['wrist_max_angle_filtered_optimal_max']),
        axis=1
    )
    merged_df['elbow_max_angle_shot_classification'] = merged_df.apply(
        lambda row: classify_joint(row['elbow_angle'], stats['elbow_max_angle_filtered_optimal_min'], stats['elbow_max_angle_filtered_optimal_max']),
        axis=1
    )
    merged_df['knee_max_angle_shot_classification'] = merged_df.apply(
        lambda row: classify_joint(row['knee_angle'], stats['knee_max_angle_filtered_optimal_min'], stats['knee_max_angle_filtered_optimal_max']),
        axis=1
    )
    merged_df.to_csv(output_path, index=False)
    print(f"Updated dataset saved to {output_path}")
    
    if debug:
        print("\nSample rows with calculated shot classifications for each joint and max angle indicators:")
        print(merged_df[['trial_id', 'wrist_angle', 'elbow_angle', 'knee_angle', 'shooting_motion', 
                        'wrist_max_angle_shot_classification', 'elbow_max_angle_shot_classification', 'knee_max_angle_shot_classification', 
                        'is_wrist_max_angle', 'is_elbow_max_angle', 'is_knee_max_angle', 
                        'elbow_max_angle_initial_optimal_max', 'elbow_max_angle_filtered_optimal_min',
                        'elbow_max_angle_filtered_optimal_max', 'knee_max_angle_initial_optimal_min',
                        'knee_max_angle_initial_optimal_max', 'knee_max_angle_filtered_optimal_min',
                        'knee_max_angle_filtered_optimal_max', 'release_knee_angle']].head())

    
    return merged_df

# Example usage in main block
if __name__ == "__main__":
    # Load data from the specified CSV file
    data_path = "../../data/processed/final_granular_dataset.csv"
    final_granular_df_with_stats = pd.read_csv(data_path)
    print(f"Loaded data from {data_path} with shape {final_granular_df_with_stats.shape}")
    print("Columns in final_granular_df_with_stats:\n", final_granular_df_with_stats.columns.tolist())
    
    final_granular_df_release_angles = calculate_release_angles(final_granular_df_with_stats, handedness='R', debug=True)
    # Example release angle usage
    final_granular_df_release_angles_ranges = calculate_optimal_release_angle_ranges(final_granular_df_release_angles, debug=True, lower_percentile=10, upper_percentile=90)

    # Assuming `final_granular_df` is predefined and loaded
    output_path = "../../data/model/shot_meter_docs/final_granular_logistic_optimized_meter_dataset.csv"  # Renamed for clarity
    final_granular_df_with_stats = calculate_optimal_max_angle_ranges(final_granular_df_release_angles_ranges, debug=True, lower_percentile=10, upper_percentile=90, output_path=output_path)
  
