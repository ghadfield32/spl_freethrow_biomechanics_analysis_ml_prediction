
import pandas as pd
import numpy as np


def _print_debug_info(df_before, df_after, new_columns, step_name, debug):
    """
    Helper function to print debug information.

    Parameters:
    - df_before (pd.DataFrame): DataFrame before processing.
    - df_after (pd.DataFrame): DataFrame after processing.
    - new_columns (list): List of new columns added.
    - step_name (str): Name of the processing step.
    - debug (bool): Flag to enable debug printing.
    """
    if debug:
        print(f"Step: {step_name}")
        print(f"DataFrame shape before: {df_before.shape}")
        print(f"DataFrame shape after: {df_after.shape}")
        if new_columns:
            print(f"New columns added: {new_columns}")
            for col in new_columns:
                dtype = df_after[col].dtype
                sample = df_after[col].dropna().unique()[:5]
                print(f" - {col}: dtype={dtype}, sample values={sample}")
        print("-" * 50)
    else:
        print(f"Step '{step_name}' completed.")


def calculate_continuous_frame_time(df, debug=False):
    """
    Calculates continuous frame time across all trials.
    Resets time for each trial and maintains a cumulative time across all trials.
    """
    step_name = "Calculating continuous frame time"
    df_before = df.copy()

    # Ensure the DataFrame is sorted by trial_id and frame_time for calculations
    df = df.sort_values(by=['trial_id', 'frame_time']).reset_index(drop=True)

    # Calculate trial-relative time
    df['by_trial_time'] = df.groupby('trial_id')['frame_time'].transform(lambda x: x - x.min())

    # Add cumulative time across trials
    trial_offsets = df.groupby('trial_id')['by_trial_time'].max().cumsum().shift(fill_value=0)
    df['continuous_frame_time'] = df['by_trial_time'] + df['trial_id'].map(trial_offsets)

    df_after = df.copy()
    new_columns = ['by_trial_time', 'continuous_frame_time']

    # Validation
    if (df['continuous_frame_time'] < 0).any():
        raise ValueError("Continuous frame time contains negative values.")

    _print_debug_info(df_before, df_after, new_columns, step_name, debug)

    return df


def initialize_first_row_power(df, power_columns, debug=False):
    """
    Sets the first row of each trial's power columns to 0 to avoid NaN values in energy calculations.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - power_columns (list): List of power column names to initialize.
    - debug (bool): Flag to enable debug printing.

    Returns:
    - pd.DataFrame: The DataFrame with initialized power columns.
    """
    step_name = "Initializing first row of power columns to 0 for each trial"
    df_before = df.copy()

    # Identify the first row index for each trial
    first_rows = df.groupby('trial_id').head(1).index.tolist()

    # Set the specified power columns to 0 for these first rows
    df.loc[first_rows, power_columns] = 0

    df_after = df.copy()
    new_columns = []  # No new columns added, only modifying existing ones

    if debug:
        added_info = {col: df_after[col].iloc[first_rows].unique().tolist() for col in power_columns}
        print(f"Step: {step_name}")
        print(f"Modified columns: {power_columns}")
        for col, vals in added_info.items():
            print(f" - {col}: sample values={vals}")
        print("-" * 50)
    else:
        print(f"Step '{step_name}' completed.")

    return df


def calculate_energy_metrics(df, power_columns, debug=False):
    """
    Calculates energy metrics for each joint and total energy per frame.
    """
    step_name = "Calculating energy metrics"
    df_before = df.copy()

    new_columns = []
    for power_col in power_columns:
        energy_col = power_col.replace('ongoing_power', 'energy')
        df[energy_col] = df[power_col] * df['dt']
        new_columns.append(energy_col)

    # Calculate total energy
    total_energy_columns = [col.replace('ongoing_power', 'energy') for col in power_columns]
    df['total_energy'] = df[total_energy_columns].sum(axis=1)
    new_columns.append('total_energy')

    df_after = df.copy()

    _print_debug_info(df_before, df_after, new_columns, step_name, debug)

    return df


def calculate_by_trial_energy(df, energy_columns, debug=False):
    """
    Calculates energy metrics (by-trial and overall) and exhaustion scores.
    """
    step_name = "Calculating by-trial energy and exhaustion scores"
    df_before = df.copy()

    new_columns = ['by_trial_energy', 'by_trial_exhaustion_score',
                   'overall_cumulative_energy', 'overall_exhaustion_score']

    # By-trial energy
    df['by_trial_energy'] = df.groupby('trial_id')['total_energy'].cumsum()
    # By-trial exhaustion score
    df['by_trial_exhaustion_score'] = (
        df.groupby('trial_id')['by_trial_energy']
        .transform(lambda x: x / x.max())
    )

    # Overall cumulative energy
    df['overall_cumulative_energy'] = df['total_energy'].cumsum()
    max_overall_cumulative_energy = df['overall_cumulative_energy'].max()
    df['overall_exhaustion_score'] = (
        df['overall_cumulative_energy'] / max_overall_cumulative_energy
    )

    df_after = df.copy()

    _print_debug_info(df_before, df_after, new_columns, step_name, debug)

    return df


def calculate_joint_energy_metrics(df, power_columns, debug=False):
    """
    Calculates per-joint by-trial energy and exhaustion scores.
    """
    step_name = "Calculating per-joint energy metrics"
    df_before = df.copy()

    new_columns = []
    for power_col in power_columns:
        energy_col = power_col.replace('ongoing_power', 'energy')
        # By-trial energy
        by_trial_col = f'{energy_col}_by_trial'
        df[by_trial_col] = df.groupby('trial_id')[energy_col].cumsum()
        new_columns.append(by_trial_col)

        # By-trial exhaustion score
        by_trial_exhaustion_col = f'{energy_col}_by_trial_exhaustion_score'
        df[by_trial_exhaustion_col] = (
            df[by_trial_col] /
            df.groupby('trial_id')[by_trial_col].transform('max')
        )
        new_columns.append(by_trial_exhaustion_col)

        # Overall cumulative energy
        overall_cumulative_col = f'{energy_col}_overall_cumulative'
        df[overall_cumulative_col] = df[energy_col].cumsum()
        new_columns.append(overall_cumulative_col)

        # Overall exhaustion score
        overall_exhaustion_col = f'{energy_col}_overall_exhaustion_score'
        df[overall_exhaustion_col] = (
            df[overall_cumulative_col] /
            df[overall_cumulative_col].max()
        )
        new_columns.append(overall_exhaustion_col)

    df_after = df.copy()

    _print_debug_info(df_before, df_after, new_columns, step_name, debug)

    return df


def validate_metrics(df, power_columns, debug=False):
    """
    Validates the calculated metrics for consistency.
    """
    step_name = "Validating metrics"
    df_before = df.copy()

    # Validation steps
    # Check continuous frame time
    if (df['continuous_frame_time'] < 0).any():
        raise ValueError("Continuous frame time contains negative values.")

    # Validate energy columns
    for power_col in power_columns:
        energy_col = power_col.replace('ongoing_power', 'energy')
        if energy_col not in df.columns:
            raise ValueError(f"Missing energy column: {energy_col}")
        if df[energy_col].isnull().any():
            raise ValueError(f"Energy column {energy_col} contains NaN values.")

    # Validate total energy calculation
    total_energy_columns = [col.replace('ongoing_power', 'energy') for col in power_columns]
    calculated_total_energy = df[total_energy_columns].sum(axis=1)
    if not calculated_total_energy.equals(df['total_energy']):
        raise ValueError("Total energy does not match the sum of individual energy columns.")

    # Validate exhaustion scores
    if not ((df['by_trial_exhaustion_score'] >= 0) & (df['by_trial_exhaustion_score'] <= 1)).all():
        raise ValueError("By-trial exhaustion scores are not normalized between 0 and 1.")
    if not ((df['overall_exhaustion_score'] >= 0) & (df['overall_exhaustion_score'] <= 1)).all():
        raise ValueError("Overall exhaustion scores are not normalized between 0 and 1.")

    # Per-joint metrics validation can be added similarly if needed

    new_columns = []  # No new columns added during validation

    _print_debug_info(df_before, df, new_columns, step_name, debug)

    return True


def main_granular_ongoing_exhaustion_pipeline(df, power_columns, debug=False):
    """
    Main pipeline to calculate and validate all metrics.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - power_columns (list): List of power column names.
    - debug (bool): Flag to enable debug printing.

    Returns:
    - pd.DataFrame: The processed DataFrame with all metrics calculated.
    """
    step_name = "Starting main pipeline"
    if debug:
        print(f"Step: {step_name}")
        print(f"Initial DataFrame shape: {df.shape}")
        print("-" * 50)
    else:
        print(f"Step '{step_name}' completed.")

    # Step 1: Calculate continuous frame time
    df = calculate_continuous_frame_time(df, debug=debug)

    # Step 2: Initialize first row of power columns to 0
    df = initialize_first_row_power(df, power_columns, debug=debug)

    # Step 3: Calculate energy metrics
    df = calculate_energy_metrics(df, power_columns, debug=debug)

    # Step 4: Calculate by-trial energy and exhaustion scores
    energy_columns = [col.replace('ongoing_power', 'energy') for col in power_columns]
    df = calculate_by_trial_energy(df, energy_columns, debug=debug)

    # Step 5: Calculate per-joint metrics
    df = calculate_joint_energy_metrics(df, power_columns, debug=debug)

    # Step 6: Validate metrics
    validate_metrics(df, power_columns, debug=debug)

    if debug:
        print("Step: Main pipeline completed successfully.")
        print(f"Final DataFrame shape: {df.shape}")
        print("-" * 50)
    else:
        print("Step 'Main pipeline' completed successfully.")

    return df


def summarize_joint_energy_by_trial(processed_df, power_columns, debug=False):
    """
    Summarizes joint energy metrics by trial for inclusion in the ML dataset.

    Args:
        processed_df (pd.DataFrame): Granular dataset containing joint energy metrics.
        power_columns (list): List of joint power columns to process.
        debug (bool): If True, prints debugging information.

    Returns:
        pd.DataFrame: Summary of joint energy metrics by trial.
    """
    step_name = "Summarizing joint energy metrics by trial"
    df_before = processed_df.copy()

    if debug:
        print(f"Step: {step_name}")
        print(f"Initial processed_df shape: {processed_df.shape}")
        print("-" * 50)

    # Replace power column names with their corresponding energy columns
    energy_columns = [col.replace('ongoing_power', 'energy') for col in power_columns]

    # Validate energy columns exist in processed_df
    missing_columns = [col for col in energy_columns if col not in processed_df.columns]
    if missing_columns:
        raise ValueError(f"Missing energy columns in processed_df: {missing_columns}")

    # Create summarized statistics for each trial
    summary = processed_df.groupby('trial_id')[energy_columns].agg(
        **{f'{col}_mean': (col, 'mean') for col in energy_columns},
        **{f'{col}_max': (col, 'max') for col in energy_columns},
        **{f'{col}_std': (col, 'std') for col in energy_columns}
    ).reset_index()

    df_after = summary.copy()
    new_columns = list(summary.columns)

    if debug:
        print(f"Summary DataFrame shape: {summary.shape}")
        print(f"Number of trials processed: {summary['trial_id'].nunique()}")
        print(f"Columns in summary DataFrame: {summary.columns.tolist()}")
        print("-" * 50)
    else:
        print(f"Step '{step_name}' completed.")

    return summary


def merge_joint_energy_with_ml_dataset(processed_df, final_ml_df, power_columns, debug=False):
    """
    Merges summarized joint energy metrics from the granular dataset into the ML dataset.

    Args:
        processed_df (pd.DataFrame): Granular dataset containing joint energy metrics.
        final_ml_df (pd.DataFrame): Machine learning dataset.
        power_columns (list): List of joint power columns to process.
        debug (bool): If True, prints debugging information.

    Returns:
        pd.DataFrame: Updated ML dataset with joint energy metrics added.
    """
    step_name = "Merging joint energy metrics into ML dataset"
    df_before = final_ml_df.copy()

    if debug:
        print(f"Step: {step_name}")
        print(f"Shape of processed_df before summarization: {processed_df.shape}")
        print(f"Shape of final_ml_df before merge: {final_ml_df.shape}")
        print("-" * 50)

    # Summarize joint energy by trial
    energy_summary = summarize_joint_energy_by_trial(processed_df, power_columns, debug=debug)

    # Identify overlapping columns between final_ml_df and energy_summary (excluding the merge key 'trial_id')
    overlapping_columns = set(final_ml_df.columns).intersection(set(energy_summary.columns)) - {'trial_id'}

    if overlapping_columns:
        if debug:
            print(f"Overlapping columns detected: {overlapping_columns}")

        # Rename overlapping columns in energy_summary
        energy_summary = energy_summary.rename(columns={col: f"{col}_summary" for col in overlapping_columns})
        if debug:
            print(f"Renamed overlapping columns in energy_summary: {list(energy_summary.columns)}")
            print("-" * 50)

    # Merge the summarized data into the ML dataset with suffixes to handle additional collisions
    try:
        pre_merge_columns = set(final_ml_df.columns)
        final_ml_df = final_ml_df.merge(energy_summary, on='trial_id', how='left', suffixes=('', '_merged'))

        # Check for duplicate columns and resolve them
        duplicated_columns = [col for col in final_ml_df.columns if col.endswith('_merged')]
        if duplicated_columns:
            if debug:
                print(f"Duplicated columns after merge: {duplicated_columns}")
            # Drop duplicated columns or handle them based on preference
            final_ml_df = final_ml_df.drop(columns=duplicated_columns)

        if debug:
            post_merge_columns = set(final_ml_df.columns)
            added_columns = post_merge_columns - pre_merge_columns
            print(f"Shape of final_ml_df after merge: {final_ml_df.shape}")
            print(f"Columns added during merge: {sorted(added_columns)}")
            print("-" * 50)
        else:
            print(f"Step '{step_name}' completed.")

    except Exception as e:
        raise RuntimeError(f"Error during merging: {e}")

    return final_ml_df


def output_dataset(dataset, filename="final_ml_dataset.csv"):
    """
    Outputs the final dataset to a file and prints a summary.

    Args:
        dataset (pd.DataFrame): The DataFrame to output.
        filename (str): The name of the output file (default: 'final_ml_dataset.csv').
    """
    step_name = "Outputting final dataset"
    print(f"[{step_name}]")

    # Save the dataset to a CSV file
    dataset.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")

    # Print a summary of the dataset
    print("Dataset Summary:")
    print(dataset.info())
    print("First few rows of the dataset:")
    print(dataset.head())
    print("-" * 50)


# Example Usage
if __name__ == "__main__":
    import logging

    # Configure logging if needed
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Define power columns for joints
    power_columns = [
        'L_ANKLE_ongoing_power', 'R_ANKLE_ongoing_power',  # Ankles
        'L_KNEE_ongoing_power', 'R_KNEE_ongoing_power',    # Knees
        'L_HIP_ongoing_power', 'R_HIP_ongoing_power',      # Hips
        'L_ELBOW_ongoing_power', 'R_ELBOW_ongoing_power',  # Elbows
        'L_WRIST_ongoing_power', 'R_WRIST_ongoing_power',  # Wrists
        'L_1STFINGER_ongoing_power', 'R_1STFINGER_ongoing_power',  # Index fingers
        'L_5THFINGER_ongoing_power', 'R_5THFINGER_ongoing_power'   # Pinky fingers
    ]

    # Assuming final_granular_df_with_optimal_release_angles and final_ml_df_with_optimal_release_angles
    # are predefined DataFrames loaded elsewhere in the code.


    # Inspect trial IDs and their counts
    trial_ids = final_granular_df_with_optimal_release_angles['trial_id'].unique()
    trial_id_counts = final_granular_df_with_optimal_release_angles['trial_id'].value_counts()
    logger.debug(f"Unique trial IDs: {trial_ids}")
    logger.debug(f"Trial ID counts: {trial_id_counts.to_dict()}")

    # Check for missing or zero 'dt' values
    logger.debug(f"Missing dt values: {final_granular_df_with_optimal_release_angles['dt'].isnull().sum()}")
    logger.debug(f"Zero dt values: {(final_granular_df_with_optimal_release_angles['dt'] == 0).sum()}")

    # Run the pipeline with debug enabled
    final_granular_df_with_energy = main_granular_ongoing_exhaustion_pipeline(
        final_granular_df_with_optimal_release_angles,
        power_columns,
        debug=False
    )
    
    # Summarize and Merge joint energy metrics into ML dataset
    final_ml_df_with_energy = merge_joint_energy_with_ml_dataset(
        final_granular_df_with_energy,
        final_ml_df_with_optimal_release_angles,
        power_columns,
        debug=False
    )

    # # Output the datasets to files
    output_dataset(final_ml_df_with_energy, filename="../../data/processed/final_ml_dataset.csv")
    output_dataset(final_granular_df_with_energy, filename="../../data/processed/final_granular_dataset.csv")
