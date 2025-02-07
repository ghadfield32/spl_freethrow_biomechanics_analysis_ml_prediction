
import os
import logging
import pandas as pd
import numpy as np  # Added numpy import for consistency
from ml.data_load_prepare.key_feature_extraction import load_player_info, get_column_definitions
from ml.data_load_prepare.main_load_and_prepare import bulk_process_directory
from ml.feature_engineering.ml_dataset_definitions import get_ml_dataset_column_definitions
from ml.feature_engineering.energy_exhaustion_metrics import (
    main_granular_ongoing_exhaustion_pipeline,
    merge_joint_energy_with_ml_dataset,
    output_dataset
)
from ml.feature_engineering.optimal_release_angle_metrics import (
    log_trial_ids,
    check_duplicates,
    create_optimal_angle_reference_table,
    add_optimized_angles_to_granular,
    aggregate_angles,
    add_optimized_angles_to_ml
)
from ml.feature_engineering.categorize_categoricals import (
    transform_features_with_bins,
    load_default_bin_config
)

def process_basketball_data(
    directory_path,
    player_info_path,
    output_ml_path,
    output_granular_path,
    power_columns=None,
    bin_config=None,  # New parameter to accept custom bin configurations
    debug=False,
    log_level=logging.INFO,
    new_data=True  # New parameter added
):
    """
    Processes basketball free throw data for feature engineering and ML dataset preparation.
    Can append to existing datasets or overwrite them based on the `new_data` flag.

    Parameters:
    - directory_path (str): Path to the data directory.
    - player_info_path (str): Path to the player information JSON file.
    - output_ml_path (str): Path to save the final ML dataset CSV.
    - output_granular_path (str): Path to save the final granular dataset CSV.
    - power_columns (list, optional): List of power column names. Defaults to predefined list.
    - bin_config (dict, optional): Custom bin configuration. If None, loads default configuration.
    - debug (bool, optional): Flag to enable debug mode. Defaults to False.
    - log_level (int, optional): Logging level. Defaults to logging.INFO.
    - new_data (bool, optional): If True, append to existing datasets. If False, overwrite. Defaults to True.

    Returns:
    - None
    """
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    if power_columns is None:
        power_columns = [
            'L_ANKLE_ongoing_power', 'R_ANKLE_ongoing_power',  # Ankles
            'L_KNEE_ongoing_power', 'R_KNEE_ongoing_power',    # Knees
            'L_HIP_ongoing_power', 'R_HIP_ongoing_power',      # Hips
            'L_ELBOW_ongoing_power', 'R_ELBOW_ongoing_power',  # Elbows
            'L_WRIST_ongoing_power', 'R_WRIST_ongoing_power',  # Wrists
            'L_1STFINGER_ongoing_power', 'R_1STFINGER_ongoing_power',  # Index fingers
            'L_5THFINGER_ongoing_power', 'R_5THFINGER_ongoing_power'   # Pinky fingers
        ]

    logger.info("Starting basketball data processing pipeline.")

    try:
        # ---------Bulk Load-----------
        logger.debug("Starting bulk processing of data directory.")
        final_granular_df, final_ml_df = bulk_process_directory(
            directory_path, player_info_path, debug=debug
        )
        logger.info("Bulk processing completed.")
        
        # Output column definitions
        logger.info("[Step: Final Granular Dataset Column Definitions]")
        column_definitions = get_column_definitions()
        for col, desc in column_definitions.items():
            logger.info(f"{col}: {desc}")
        
        # Output ML Dataset Column Definitions
        logger.debug("Retrieving ML dataset column definitions.")
        column_definitions_ml = get_ml_dataset_column_definitions()
        logger.info("ML Dataset Column Definitions:")
        for col, desc in column_definitions_ml.items():
            logger.info(f"{col}: {desc}")

        #------------Feature Engineer----------
        logger.debug("Starting feature engineering.")

        # Log trial IDs before processing
        log_trial_ids(final_granular_df, "Initial Load - Granular DF", debug=debug)
        log_trial_ids(final_ml_df, "Initial Load - ML DF", debug=debug)

        # Check for duplicates in final_ml_df
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
        logger.info("Optimized release angles added to ML dataset.")

        # Run the energy exhaustion pipeline
        logger.debug("Running energy exhaustion pipeline.")
        final_granular_df_with_energy = main_granular_ongoing_exhaustion_pipeline(
            final_granular_df_with_optimal_release_angles,
            power_columns,
            debug=debug
        )
        logger.info("Energy exhaustion metrics computed.")

        # Merge energy metrics into ML dataset
        final_ml_df_with_energy = merge_joint_energy_with_ml_dataset(
            final_granular_df_with_energy,
            final_ml_df_with_optimal_release_angles,
            power_columns,
            debug=debug
        )
        logger.info("Energy metrics merged into ML dataset.")

        #-------------Categoricals Handling-------------
        logger.debug("Starting categoricals handling.")

        # Load bin configuration
        if bin_config is None:
            logger.debug("Loading default bin configuration.")
            bin_config = load_default_bin_config()
        else:
            logger.debug("Using provided bin configuration.")

        # Transform player features using the configuration
        categorized_columns_df = transform_features_with_bins(final_ml_df_with_energy, bin_config, debug=debug)

        # Combine the original ML DataFrame with the categorized columns
        final_ml_df_categoricals = pd.concat([final_ml_df_with_energy, categorized_columns_df], axis=1)

        # Debugging output
        if debug:
            logger.debug("\nFinal DataFrame with Categorized Features:")
            logger.debug(final_ml_df_categoricals.head())
            logger.debug("Categorized Columns:")
            logger.debug(final_ml_df_categoricals.columns)

        # ---------Handle Appending or Overwriting----------
        if new_data:
            logger.info("Appending new data to existing datasets.")

            # Handle ML Dataset
            if os.path.exists(output_ml_path):
                logger.debug(f"Loading existing ML dataset from {output_ml_path}.")
                existing_ml_df = pd.read_csv(output_ml_path)
                combined_ml_df = pd.concat([existing_ml_df, final_ml_df_categoricals], ignore_index=True)
                combined_ml_df.drop_duplicates(inplace=True)
                logger.info("Appended new data to ML dataset and removed duplicates.")
            else:
                logger.warning(f"ML dataset file {output_ml_path} does not exist. Creating a new one.")
                combined_ml_df = final_ml_df_categoricals.copy()

            # Handle Granular Dataset
            if os.path.exists(output_granular_path):
                logger.debug(f"Loading existing granular dataset from {output_granular_path}.")
                existing_granular_df = pd.read_csv(output_granular_path)
                combined_granular_df = pd.concat([existing_granular_df, final_granular_df_with_energy], ignore_index=True)
                combined_granular_df.drop_duplicates(inplace=True)
                logger.info("Appended new data to granular dataset and removed duplicates.")
            else:
                logger.warning(f"Granular dataset file {output_granular_path} does not exist. Creating a new one.")
                combined_granular_df = final_granular_df_with_energy.copy()

            # Output the combined datasets to files
            output_dataset(combined_ml_df, filename=output_ml_path)
            output_dataset(combined_granular_df, filename=output_granular_path)
            logger.info(f"Final datasets appended and saved to {output_ml_path} and {output_granular_path}.")

        else:
            logger.info("Overwriting existing datasets with new data.")

            # Output the datasets to files, overwriting existing ones
            output_dataset(final_ml_df_categoricals, filename=output_ml_path)
            output_dataset(final_granular_df_with_energy, filename=output_granular_path)
            logger.info(f"Final datasets overwritten and saved to {output_ml_path} and {output_granular_path}.")

        logger.info("Basketball data processing pipeline completed successfully.")

    except Exception as e:
        logger.exception("An error occurred during the processing pipeline.")
        raise e

if __name__ == "__main__":
    import logging

    # Define paths
    directory_path = "../../../SPL-Open-Data/basketball/freethrow/data/P0001"
    player_info_path = "../../../SPL-Open-Data/basketball/freethrow/participant_information.json"
    output_ml_path = "../../../data/processed/final_ml_dataset.csv"
    output_granular_path = "../../../data/processed/final_granular_dataset.csv"

    # Optional: Define a new bin configuration if needed
    # new_bin_config = {
    #     'player_height_in_meters': {
    #         'bins': [0, 1.75, 1.95, np.inf],
    #         'labels': ["Short", "Medium", "Tall"]
    #     },
    #     # Add or modify other columns as needed
    # }

    # Call the processing function
    process_basketball_data(
        directory_path=directory_path,
        player_info_path=player_info_path,
        output_ml_path=output_ml_path,
        output_granular_path=output_granular_path,
        debug=True,  # Enable debug mode for detailed logs
        log_level=logging.DEBUG,  # Set logging level to DEBUG
        new_data=False,  # Set to True to append, False to overwrite
        bin_config=None  # Pass `new_bin_config` here if using a custom configuration
    )
