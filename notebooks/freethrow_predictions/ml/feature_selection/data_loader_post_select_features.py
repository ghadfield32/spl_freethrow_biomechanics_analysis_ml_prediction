
import logging
import pandas as pd
import pickle
from typing import Optional, Dict


def load_feature_names_for_base_data(filepath: str):
    """
    Load feature names from a pickle file.
    """
    with open(filepath, 'rb') as file:
        feature_names = pickle.load(file)
    return feature_names


def load_base_data_for_dataset(filepath: str):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(filepath)


def filter_base_data_for_select_features(dataset: pd.DataFrame, feature_names: list, debug: bool = False):
    """
    Filter the dataset to include only the specified feature names.
    """
    if feature_names is not None and len(feature_names) > 0:
        # Ensure only columns present in both the DataFrame and the selected features list are retained
        common_columns = set(dataset.columns).intersection(feature_names)
        filtered_dataset = dataset[list(common_columns)]
        if debug:
            print("Loaded and filtered dataset based on selected features:")
            print(filtered_dataset.head())
        return filtered_dataset
    else:
        print("No valid selected features found.")
        return None


def load_selected_features_data(
    features_path: str,
    dataset_path: str,
    y_variable: str,
    debug: bool = False
) -> pd.DataFrame:
    """
    Process machine learning data.

    Args:
        features_path (str): Path to the file containing feature names.
        dataset_path (str): Path to the main dataset file.
        y_variable (str): The target variable name.
        debug (bool): Flag to enable detailed debugging information.

    Returns:
        pd.DataFrame: The processed dataset ready for further analysis.

    Raises:
        ValueError: If any required step fails or invalid input is provided.
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    try:
        # Load the list of selected feature names
        logging.info("Loading selected features...")
        selected_features = load_feature_names_for_base_data(features_path)

        # Load the dataset
        logging.info("Loading dataset...")
        final_ml_df = load_base_data_for_dataset(dataset_path)

        # Filter the DataFrame using the loaded list of selected feature names
        logging.info("Filtering dataset for selected features...")
        final_ml_df_selected_features = filter_base_data_for_select_features(
            final_ml_df, 
            selected_features, 
            debug=debug
        )

        if final_ml_df_selected_features is None or final_ml_df_selected_features.empty:
            raise ValueError("Filtered DataFrame is empty or invalid.")
        
        logging.info("Data processing complete. Returning processed DataFrame.")
        return final_ml_df_selected_features

    except Exception as e:
        logging.error(f"An error occurred during data processing: {e}")
        raise


if __name__ == "__main__":
    # Example usage:
    final_ml_df_selected_features = load_selected_features_data(
        features_path='../../data/model/pipeline/final_ml_df_selected_features_columns.pkl',
        dataset_path='../../data/processed/final_ml_dataset.csv',
        y_variable='result',
        debug=True
    )
